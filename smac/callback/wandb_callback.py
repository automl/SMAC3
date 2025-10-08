from __future__ import annotations

import smac
from smac.callback import Callback


class WandBCallback(Callback):
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
            **kwargs
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
            **kwargs
        )
        super().__init__()


    def on_end(self, smbo: smac.main.smbo.SMBO) -> None:
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



