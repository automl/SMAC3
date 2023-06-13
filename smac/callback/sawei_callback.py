from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import trim_mean
import smac
from smac.acquisition.function import LCB, UCB
from smac.acquisition.maximizer import (
    AbstractAcquisitionMaximizer,
    LocalAndSortedRandomSearch,
)
from smac.callback import Callback
from smac.main.smbo import SMBO
from smac.model.random_forest import RandomForest
from smac.model.gaussian_process import GaussianProcess
from smac.runhistory import TrialInfo, TrialKey, TrialValue
from ConfigSpace import Configuration
from smac.model import AbstractModel
from smac.acquisition.function import WEI
import smac
from smac import Callback
from smac.acquisition.function import EI
import numpy as np
from smac.acquisition.function import WEI
from smac.callback.utils import query_callback
from pathlib import Path
from smac.utils.logging import get_logger
from smac.runhistory.encoder.encoder import RunHistoryEncoder

logger = get_logger(__name__)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

import wandb



class UpperBoundRegretCallback(Callback):
    def __init__(
            self, 
            top_p: float = 0.5, 
            ucb: UCB | None = None, 
            lcb: LCB | None = None
        ) -> None:
        super().__init__()

        # Use only top p portion of the evaluated configs to fit the model (Sec. 4)
        # DISCUSS: What does top p configs mean for AC?
        self.top_p: float = top_p  
        self.ubr: float | None = None
        self.history: list[dict[str, Any]] = []
        self._UCB: UCB = ucb or UCB()
        self._LCB: LCB = lcb or LCB()

    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        # Check encoding
        if type(smbo.intensifier.config_selector._runhistory_encoder) is not RunHistoryEncoder:
            msg = "Currently no response value transformations are supported, only "\
                  "RunHistoryEncoder possible."
            # raise NotImplementedError(msg)
                
        # Line 16: r_t = min UCB(config) (from all evaluated configs) - min LCB(config) (from config space)
        # Get all evaluated configs
        rh = smbo.runhistory
        evaluated_configs = rh.get_configs(sort_by="cost")
        evaluated_configs = evaluated_configs[:int(np.ceil(len(evaluated_configs) * self.top_p))]

        # Prepare acquisition functions
        model = smbo.intensifier.config_selector._model
        # BUG: num data is calculated wrongly
        # calculate UBR right from the start, filter to sbo if necessary
        if model_fitted(model): 
            kwargs = {"model": model, "num_data": rh.finished}
            self._UCB.update(**kwargs)
            self._LCB.update(**kwargs)

            # Minimize UCB (max -UCB) for all evaluated configs
            acq_values = self._UCB(evaluated_configs)
            min_ucb =  -float(np.squeeze(np.amax(acq_values)))

            # Minimize LCB (max -LCB) on config space
            acq_maximizer = LocalAndSortedRandomSearch(
                configspace=smbo._configspace,
                seed=smbo.scenario.seed,
                acquisition_function=self._LCB,
            )
            challengers = acq_maximizer._maximize(
                previous_configs=[],
                n_points=1,
            )
            challengers = np.array(challengers, dtype=object)
            acq_values = challengers[:, 0]
            min_lcb = -float(np.squeeze(np.amax(acq_values)))

            # TODO log transform (rh encoder)
            # feature/_stopping_callback

            self.ubr = min_ucb - min_lcb

            info = {
                "n_evaluated": smbo.runhistory.finished,
                "ubr": self.ubr,
                "min_ucb": min_ucb,
                "min_lcb": min_lcb,
            }

            logger.debug(f"Upper Bound Regret: n={smbo.runhistory.finished}, " + ", ".join([f"{k}={v:.4f}" for k, v in info.items() if k != "n_evaluated"]))

            self.history.append(info)

        return super().on_tell_end(smbo, info, value)
    
    def on_end(self, smbo: smac.main.smbo.SMBO) -> None:
        # Write history
        path = smbo._scenario.output_directory
        if path is not None:
            df = pd.DataFrame(data=self.history)
            fn = Path(path) / "ubr_history.csv"
            df.to_csv(fn, header=True, index=False)
        return super().on_end(smbo)


class WEITracker(Callback):
    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []
        super().__init__()

    def on_next_configurations_end(self, config_selector: smac.main.config_selector.ConfigSelector, config: Configuration) -> None:
        model = config_selector._model
        if issubclass(type(config_selector._acquisition_function), WEI) and model_fitted(model):  # FIXME: Flag _is_trained only exists for GP so far:
            X = config.get_array()
            # TODO: pipe X through
            acq_values = config_selector._acquisition_function([config])
            alpha = config_selector._acquisition_function._alpha
            pi_term = config_selector._acquisition_function.pi_term[0][0]
            ei_term = config_selector._acquisition_function.ei_term[0][0]

            info = {
                "n_evaluated": config_selector._runhistory.finished,
                "alpha": alpha,
                "pi_term": pi_term,
                "ei_term": ei_term,
                "pi_pure_term": config_selector._acquisition_function.pi_pure_term[0][0],
                "pi_mod_term": config_selector._acquisition_function.pi_mod_term[0][0],
            }
            self.history.append(info)
            logger.debug(f"WEI: n={info['n_evaluated']}, " + ", ".join([f"{k}={v:.4f}" for k, v in info.items() if k != "n_evaluated"]))
        return super().on_next_configurations_end(config_selector, config)

    def on_end(self, smbo: smac.main.smbo.SMBO) -> None:
        # Write history
        path = smbo._scenario.output_directory
        if path is not None:
            df = pd.DataFrame(data=self.history)
            fn = Path(path) / "wei_history.csv"
            df.to_csv(fn, header=True, index=False)
        return super().on_end(smbo)



def detect_switch(UBR: np.array, window_size: int = 10, atol_rel: float = 0.1) -> np.array[bool]:
    miqm = apply_moving_iqm(U=UBR, window_size=window_size)
    miqm_gradient = np.gradient(miqm)

    # max_grad = np.maximum.accumulate(miqm_gradient)
    # switch = np.array([np.isclose(miqm_gradient[i], 0, atol=atol_rel*max_grad[i]) for i in range(len(miqm_gradient))])
    # switch[0] = 0  # misleading signal bc of iqm

    G_abs = np.abs(miqm_gradient)
    max_grad = [np.nanmax(G_abs[:i+1]) for i in range(len(G_abs))]
    switch = np.array([np.isclose(miqm_gradient[i], 0, atol=atol_rel*max_grad[i]) for i in range(len(miqm_gradient))])
    # switch = np.isclose(miqm_gradient, 0, atol=1e-5)
    switch[:window_size] = 0  # misleading signal bc of iqm
    
    return switch

# Moving IQM
def apply_moving_iqm(U: np.array, window_size: int = 5) -> np.array:

    def moving_iqm(X: np.array) -> float:
        return trim_mean(X, 0.25)

    U_padded = np.concatenate((np.array([U[0]] * (window_size - 1)), U))
    slices = sliding_window_view(U_padded, window_size)
    miqm = np.array([moving_iqm(s) for s in slices])
    return miqm


def model_fitted(model: AbstractModel) -> bool:
    return (type(model) == GaussianProcess and model._is_trained) or (type(model) == RandomForest and model._rf is not None)


class SAWEI(Callback):
    def __init__(
        self, 
        alpha: float = 0.5, 
        delta: float | str = 0.1,
        window_size: int = 7,
        atol_rel: float = 0.1,
        track_attitude: str = "last",
        use_pure_PI: bool = False,
        auto_alpha: bool = False,
    ) -> None:
        # alpha = 1: PI = Exploiting
        # alpha = 0.5: EI = Exploring/Balance
        # alpha = 0: Exploring
        self.alpha = alpha
        self.delta = delta
        self.window_size = window_size
        self.atol_rel = atol_rel
        self.track_attitude = track_attitude
        self.use_pure_PI = use_pure_PI
        self.auto_alpha = auto_alpha

        if self.delta == "auto":
            self.auto_alpha = True

        self.last_inc_count: int = 0
        self._pi_term_sum: float = 0.
        self._ei_term_sum: float = 0.
        self.bounds = (0., 1.)
        self.history: list[dict[str, Any]] = []

        self.wandb_run = None
        self.wandb_run = wandb.init(
            project="sawei",
            job_type="dev",
            entity="benjamc",
            group="dev",
            dir="./tmp/sawei",
        )

    def __str__(self) -> str:
        return "Self-Adjusting Weighted Expected Improvement (SAWEI)"

    def on_tell_end(self, solver: smac.main.smbo.SMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        model = solver.intensifier.config_selector._model
        if model_fitted(model):
            state = {
                "n_evaluated": solver.runhistory.finished,
                "alpha": query_callback(solver=solver, callback_type="WEITracker", key="alpha"),
                "n_incumbent_changes": solver._intensifier._incumbents_changed,
                "wei_ei_term": query_callback(solver=solver, callback_type="WEITracker", key="ei_term"),
                "wei_pi_pure_term": query_callback(solver=solver, callback_type="WEITracker", key="pi_pure_term"),
                "wei_pi_mod_term": query_callback(solver=solver, callback_type="WEITracker", key="pi_mod_term"),
                "ubr": query_callback(solver=solver, callback_type="UpperBoundRegretCallback", key="ubr"),
            }
            self.history.append(state)
            # Check if it is time to switch
            switch = False
            UBR = [s["ubr"] for s in self.history]

            if self.use_pure_PI:
                key_pi = "wei_pi_pure_term"
            else:
                key_pi = "wei_pi_mod_term"

            # We need at least 2 UBRs to compute the gradient
            if len(UBR) >= 2: 
                switch = detect_switch(UBR=UBR, window_size=self.window_size, atol_rel=self.atol_rel)[-1]

            self._pi_term_sum += state[key_pi]
            self._ei_term_sum += state["wei_ei_term"]

            if switch:
                if self.track_attitude == "last":
                    # Calculate attitude: Exploring or exploiting?
                    # Exploring = when ei term is bigger
                    # Exploiting = when pi term is bigger
                    exploring = state[key_pi] <= state["wei_ei_term"]
                    distance = state["wei_ei_term"] - state[key_pi]
                elif self.track_attitude in ["until_inc_change", "until_last_switch"]:
                    exploring =  self._pi_term_sum <= self._ei_term_sum
                    distance = self._ei_term_sum - self._pi_term_sum
                else:
                    raise ValueError(f"Unknown track_attitude {self.track_attitude}.")

                if self.auto_alpha:
                    alpha = sigmoid(distance)
                else:
                    # If attitude is
                    # - exploring (exploring==True): increase alpha, change to exploiting
                    # - exploiting (exploring==False): decrease alpha, change to exploring
                    sign = 1 if exploring else -1
                    alpha = self.alpha + sign * self.delta

                # Bound alpha
                lb, ub = self.bounds
                self.alpha = max(lb, min(ub, alpha))

            if self.track_attitude == "until_inc_change":
                if state["n_incumbent_changes"] > self.last_inc_count:
                    self.last_inc_count = state["n_incumbent_changes"]
                    self._pi_term_sum: float = 0.
                    self._ei_term_sum: float = 0.
            elif self.track_attitude == "until_last_switch":
                if switch:
                    self._pi_term_sum: float = 0.
                    self._ei_term_sum: float = 0.

            if type(solver.intensifier._config_selector._acquisition_function) == WEI:
                self.modify_solver(solver=solver, alpha=self.alpha)

            info = {
                "switch": int(switch),
            }
            state.update(info)

            if self.wandb_run:
                self.wandb_run.log(data=state)

        return super().on_tell_end(solver, info, value)
    
    def modify_solver(self, solver: smac.main.smbo.SMBO, alpha: int | float | None) -> smac.main.smbo.SMBO:
        if alpha is not None:
            # if self.discrete_actions is not None:
            #     action = self.discrete_actions[action]
            alpha = float(alpha)

            # if not (self.action_bounds[0] <= action <= self.action_bounds[1]):
            #     raise ValueError(f"Action (xi) is '{action}' but only is allowed in range '{self.action_bounds}'.")

            kwargs = {
                "eta": solver.runhistory.get_cost(solver.intensifier.get_incumbent()), 
                "alpha": alpha,
                "num_data": solver.runhistory.finished,
            }
            solver.intensifier.config_selector._acquisition_function._update(**kwargs)

    def on_end(self, smbo: smac.main.smbo.SMBO) -> None:
        if self.wandb_run:
            self.wandb_run.finish()
        return super().on_end(smbo)


def get_sawei_kwargs(
    ubr_top_p: float = 0.5,
    sawei_alpha: float = 0.5, 
    sawei_delta: float | str = 0.1,
    sawei_window_size: int = 7,
    sawei_atol_rel: float = 0.1,
    sawei_track_attitude: str = "last",
    sawei_use_pure_PI: bool = False,
    sawei_auto_alpha: bool = False,
):
    # TODO docstring
    # TODO fix warnings
    # TODO create tests
    # TODO set logging dir of ubr and weitracker
    ubr = UpperBoundRegretCallback(top_p=ubr_top_p)
    weitracker = WEITracker()
    sawei = SAWEI(
        alpha=sawei_alpha,
        delta=sawei_delta,
        window_size=sawei_window_size,
        atol_rel=sawei_atol_rel,
        track_attitude=sawei_track_attitude,
        use_pure_PI=sawei_use_pure_PI,
        auto_alpha=sawei_auto_alpha,
    )

    callbacks = [ubr, weitracker, sawei]
    acquisition_function = WEI()

    kwargs = {
        "callbacks": callbacks,
        "acquisition_function": acquisition_function
    }

    return kwargs
