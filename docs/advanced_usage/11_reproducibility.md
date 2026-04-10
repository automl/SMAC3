# Reproducibility

Reproducibility can only be ensured if one worker is used and no time (wallclock or CPU time) is involved.

!!! warning
    SMBO.reset() will not seed smac with the original seed. If you want to have a full reset, please set the seed again after calling reset.