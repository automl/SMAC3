from typing import Any

from dataclasses import asdict

import smac
from smac.callback import Callback
from smac.runhistory import TrialInfo, TrialValue


class WandBCallback(Callback):
    """

    Callback to log the results of the optimization as well as intermediate information to WandB.

    Logs TrialInfo, TrialValue and the number of successfully executed trials (as step) to WandB `on_tell_end`.
    Upon the end of the run, logs the trajectory of the intensifier to WandB.

    Parameters
    ----------
    project : str
        The project name of the WandB project.
    entity : str
        The entity name of the WandB project.
    id : str, optional
        The id of the run.
    outdir : str, optional
        The output directory of the WandB run.
    mode : str, optional
        The mode of the WandB run.
    resume : str, optional
        The resume mode of the WandB run.
    job_type : str, optional
        The job type of the WandB run.
    group : str, optional
        The group of the WandB run.
    config : dict or str, optional
        The configuration of the WandB run.
    save_code : bool, optional
        Whether to save the code of the WandB run.
    **kwargs : dict
        Additional arguments to pass to the WandB run.
    """

    def __init__(
        self,
        project: str,
        entity: str,
        id: str | None = None,
        outdir: str | None = None,
        mode: str | None = None,
        resume: str = "allow",
        job_type: str | None = None,
        group: str | None = None,
        config: dict | str | None = None,
        save_code: bool = True,
        **kwargs: dict[str, Any],
    ) -> None:
        import wandb

        self.run = wandb.init(
            id=id,
            resume=resume,
            mode=mode,
            project=project,
            job_type=job_type,
            entity=entity,
            group=group,
            dir=outdir,
            config=config,
            save_code=save_code,
            **kwargs,
        )
        super().__init__()

    def on_tell_end(self, smbo: smac.main.smbo.SMBO, info: TrialInfo, value: TrialValue) -> bool | None:  # noqa: D102
        info_dict = asdict(info)
        info_dict["config"] = info_dict["config"].get_dictionary()
        value_dict = asdict(value)
        log_dict = info_dict | value_dict
        log_dict["step"] = smbo.runhistory.finished
        self.run.log(data=log_dict)
        return super().on_tell_end(smbo, info, value)

    def on_end(self, smbo: smac.main.smbo.SMBO) -> None:  # noqa: D102
        intensifier_data = smbo.intensifier.get_data()
        trajectory = intensifier_data["trajectory"]
        import pandas as pd

        df = pd.DataFrame(data=trajectory)
        print(df)
        # trajectory = Table(dataframe=df, allow_mixed_types=True)
        df["costs"] = df["costs"].apply(lambda x: x[0])  # TODO properly log multi costs
        for index, row in df.iterrows():
            print(dict(row))
            self.run.log(dict(row))
        self.run.finish()
        return super().on_end(smbo)
