# Claude-Opus-5-Max

Model `claude-opus-5` through GitHub Copilot CAPI, `max` reasoning effort, adaptive thinking and default sampling.

BALROG naive agent; text history 16, image history 0; 8,192 output tokens
including reasoning. Default tasks, horizons and native random seeds
(recorded in each episode); native resume for interrupted allocations.
BALROG revision: `b7afe79e3e4265811cfa985ed7c95c4d1a11e3f5`.

Overall progress: **63.40% ± 1.85 percentage points**
(standard error), from upstream `submit.py`; 254 completed episodes.

| Environment | Episodes | Progress |
|---|---:|---:|
| babyai | 50 | 100.00% |
| babaisai | 120 | 95.83% |
| textworld | 30 | 71.57% |
| crafter | 10 | 68.18% |
| nle | 4 | 7.30% |
| minihack | 40 | 37.50% |

NetHack uses 4 of 5 planned seeds; all other environments are complete.
