import unittest

import numpy as np
import torch
from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    UniformFloatHyperparameter,
)
from gpytorch.constraints.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.priors import HorseshoePrior, LogNormalPrior

from smac.epm.gaussian_process.augmented import GloballyAugmentedLocalGaussianProcess
from smac.epm.random_forest.rf_with_instances import RandomForestWithInstances
from smac.epm.utils import check_subspace_points, get_types
from smac.facade.smac_bb_facade import SMAC4BB
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.optimizer.configuration_chooser.boing_chooser import (
    BOinGChooser,
    subspace_extraction,
)
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm_boing import RunHistory2EPM4ScaledLogCostWithRaw
from smac.scenario.scenario import Scenario
from smac.tae import StatusType
from smac.utils import test_helpers


class TestEPMChooserBOinG(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario(
            {"cs": test_helpers.get_branin_config_space(), "run_obj": "quality", "output_dir": "data-test_epmchooser"}
        )
        self.output_dirs = []
        self.output_dirs.append(self.scenario.output_dir)

        exp_kernel = MaternKernel(
            2.5,
            lengthscale_constraint=Interval(
                torch.tensor(np.exp(-6.754111155189306).repeat(2)),
                torch.tensor(np.exp(0.0858637988771976).repeat(2)),
                transform=None,
                initial_value=1.0,
            ),
            ard_num_dims=2,
            active_dims=(0, 1),
        ).double()

        noise_prior = HorseshoePrior(0.1)
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior, noise_constraint=Interval(np.exp(-25), np.exp(2), transform=None)
        ).double()

        kernel = ScaleKernel(
            exp_kernel,
            outputscale_constraint=Interval(np.exp(-10.0), np.exp(2.0), transform=None, initial_value=2.0),
            outputscale_prior=LogNormalPrior(0.0, 1.0),
        )

        self.model_kwargs = dict(kernel=kernel, likelihood=likelihood)

    def test_init(self):
        seed = 42
        config = self.scenario.cs.sample_configuration()
        rh = RunHistory()
        rh.add(config, 10, 10, StatusType.SUCCESS)

        epm_chooser_kwargs = {
            "model_local": GloballyAugmentedLocalGaussianProcess,
            "model_local_kwargs": self.model_kwargs,
        }

        smbo_kwargs = {"epm_chooser": BOinGChooser, "epm_chooser_kwargs": epm_chooser_kwargs}

        self.assertRaisesRegex(
            ValueError,
            "BOinG only supports RandomForestWithInstances as its global optimizer",
            SMAC4BB,
            scenario=self.scenario,
            rng=seed,
            runhistory=rh,
            smbo_kwargs=smbo_kwargs,
            runhistory2epm=RunHistory2EPM4ScaledLogCostWithRaw,
        )
        self.assertRaisesRegex(
            ValueError,
            "BOinG only supports RunHistory2EPM4CostWithRaw as its rh transformer",
            SMAC4HPO,
            scenario=self.scenario,
            rng=seed,
            runhistory=rh,
            smbo_kwargs=smbo_kwargs,
        )

        epm_chooser = SMAC4HPO(
            scenario=self.scenario,
            rng=seed,
            runhistory=rh,
            smbo_kwargs=smbo_kwargs,
            runhistory2epm=RunHistory2EPM4ScaledLogCostWithRaw,
        ).solver.epm_chooser
        self.assertFalse(hasattr(epm_chooser, "turbo_optimizer"))

        epm_chooser_kwargs.update({"do_switching": True})
        epm_chooser = SMAC4HPO(
            scenario=self.scenario,
            rng=seed,
            runhistory=rh,
            smbo_kwargs=smbo_kwargs,
            runhistory2epm=RunHistory2EPM4ScaledLogCostWithRaw,
        ).solver.epm_chooser
        self.assertTrue(hasattr(epm_chooser, "turbo_optimizer"))

    def test_choose_next(self):
        seed = 42
        config = self.scenario.cs.sample_configuration()
        rh = RunHistory()
        rh.add(config, 10, 10, StatusType.SUCCESS)

        epm_chooser_kwargs = {
            "model_local": GloballyAugmentedLocalGaussianProcess,
            "model_local_kwargs": self.model_kwargs,
        }

        smbo_kwargs = {"epm_chooser": BOinGChooser, "epm_chooser_kwargs": epm_chooser_kwargs}

        epm_chooser = SMAC4HPO(
            scenario=self.scenario,
            rng=seed,
            runhistory=rh,
            smbo_kwargs=smbo_kwargs,
            runhistory2epm=RunHistory2EPM4ScaledLogCostWithRaw,
        ).solver.epm_chooser
        x = next(epm_chooser.choose_next())
        # when number of points is not large enough for building a subspace, GP works locally
        self.assertEqual(x.origin, "Local Search")
        for i in range(15):
            config = self.scenario.cs.sample_configuration()
            rh.add(config, 10, 10, StatusType.SUCCESS)

        x = next(epm_chooser.choose_next())
        # when number of points is already large enough for building a subspace, BOinG takes over
        self.assertEqual(x.origin, "BOinG")

        epm_chooser_kwargs.update({"do_switching": True})
        epm_chooser = SMAC4HPO(
            scenario=self.scenario,
            rng=seed,
            runhistory=rh,
            smbo_kwargs=smbo_kwargs,
            runhistory2epm=RunHistory2EPM4ScaledLogCostWithRaw,
        ).solver.epm_chooser
        epm_chooser.run_TuRBO = True
        x = next(epm_chooser.choose_next())
        self.assertEqual(x.origin, "TuRBO")

    def test_do_switching(self):
        seed = 42

        config = self.scenario.cs.sample_configuration()
        rh = RunHistory()
        rh.add(config, 10, 10, StatusType.SUCCESS)

        epm_chooser_kwargs = {
            "model_local": GloballyAugmentedLocalGaussianProcess,
            "model_local_kwargs": self.model_kwargs,
            "do_switching": True,
        }
        turbo_kwargs = {"failure_tol_min": 1, "length_min": 0.6}
        epm_chooser_kwargs.update({"turbo_kwargs": turbo_kwargs})

        smbo_kwargs = {"epm_chooser": BOinGChooser, "epm_chooser_kwargs": epm_chooser_kwargs}

        epm_chooser = SMAC4HPO(
            scenario=self.scenario,
            rng=seed,
            runhistory=rh,
            smbo_kwargs=smbo_kwargs,
            runhistory2epm=RunHistory2EPM4ScaledLogCostWithRaw,
        ).solver.epm_chooser

        for i in range(15):
            config = self.scenario.cs.sample_configuration()
            rh.add(config, 10, 10, StatusType.SUCCESS)
        config = self.scenario.cs.sample_configuration()
        # ensure config is the incumbent
        rh.add(config, 9.99, 10, StatusType.SUCCESS)
        next(epm_chooser.choose_next())

        # init an optimal config
        np.testing.assert_allclose(config.get_array(), epm_chooser.optimal_config)
        self.assertAlmostEqual(9.99, epm_chooser.optimal_value)
        self.assertEqual(0, epm_chooser.failcount_BOinG)

        epm_chooser.failcount_BOinG = 19
        # in this case, prob_to_TurBO becomes 1
        with unittest.mock.patch(
            "smac.optimizer.configuration_chooser.boing_chooser.BOinGChooser." "restart_TuRBOinG"
        ) as mk:
            next(epm_chooser.choose_next())
            self.assertTrue(epm_chooser.run_TuRBO)
            self.assertTrue(mk.called)

        # switch to TuRBO
        for i in range(1000):
            next(epm_chooser.choose_next())
            if not epm_chooser.run_TuRBO:
                break
        # TuRBO will be replaced with BOinG if it cannot find a better value conintuously
        self.assertLess(i, 999)

        epm_chooser.failcount_BOinG = 19
        next(epm_chooser.choose_next())

        config = self.scenario.cs.sample_configuration()
        rh.add(config, 9.5, 10, StatusType.SUCCESS)
        epm_chooser.turbo_optimizer.init_configs = []

        for i in range(10):
            next(epm_chooser.choose_next())
            if not epm_chooser.run_TuRBO:
                break
        # one time success and two times failure totally 3 times evaluations and in this case we have i==2
        self.assertEqual(i, 2)


class TestSubSpaceExtraction(unittest.TestCase):
    def test_subspace_extraction(self):
        cs = ConfigurationSpace(0)
        cs.add_hyperparameter(UniformFloatHyperparameter("x0", 0.0, 1.0))
        cs.add_hyperparameter(CategoricalHyperparameter("x1", [0, 1, 2, 3, 4, 5]))

        types, bounds = get_types(cs)
        rf = RandomForestWithInstances(
            cs,
            types=types,
            bounds=bounds,
            seed=0,
            num_trees=10,
            ratio_features=1.0,
            min_samples_split=2,
            min_samples_leaf=1,
        )

        X = np.array([[0.0, 0], [0.2, 1], [0.3, 2], [0.7, 5], [0.6, 3]])

        Y = np.array([0.1, 0.2, 0.7, 0.6, 0.5])

        X_inc = np.array([0.4, 3])
        rf.train(X, Y)

        ss_extraction_kwargs = dict(X=X, challenger=X_inc, model=rf, bounds=bounds, cat_dims=[1], cont_dims=[0])

        num_min = 2
        num_max = 5

        ss_bounds_cont, ss_bounds_cat, ss_indices = subspace_extraction(
            num_min=num_min, num_max=np.inf, **ss_extraction_kwargs
        )
        self.assertTrue(num_min <= sum(ss_indices))
        x_in_ss = check_subspace_points(X_inc, [0], [1], ss_bounds_cont, ss_bounds_cat)
        self.assertTrue(x_in_ss[0])
        ss_indices_re_exam = check_subspace_points(X, [0], [1], ss_bounds_cont, ss_bounds_cat)
        self.assertEqual(sum(ss_indices), sum(ss_indices_re_exam))

        ss_bounds_cont, ss_bounds_cat, ss_indices = subspace_extraction(
            num_min=num_min, num_max=num_max, **ss_extraction_kwargs
        )
        self.assertTrue(num_min <= sum(ss_indices) <= num_max)
        x_in_ss = check_subspace_points(X_inc, [0], [1], ss_bounds_cont, ss_bounds_cat)
        self.assertTrue(x_in_ss[0])
        ss_indices_re_exam = check_subspace_points(X, [0], [1], ss_bounds_cont, ss_bounds_cat)
        self.assertEqual(sum(ss_indices), sum(ss_indices_re_exam))

        num_max = 3
        ss_bounds_cont, ss_bounds_cat, ss_indices = subspace_extraction(
            num_min=num_min, num_max=num_max, **ss_extraction_kwargs
        )
        self.assertTrue(num_min <= sum(ss_indices) <= num_max)
        self.assertTrue(x_in_ss[0])
        ss_indices_re_exam = check_subspace_points(X, [0], [1], ss_bounds_cont, ss_bounds_cat)
        self.assertEqual(sum(ss_indices), sum(ss_indices_re_exam))
