[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/bigpig-collab/bigpig-gta6-1/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/bigpig-collab/bigpig-gta6-1/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# gta6-twitch-ccu-forecast-01 — entry v1

Forecasts Grand Theft Auto VI's launch-day Twitch concurrent viewers, hours
2–24 (UTC), as five quantiles per hour. Python 3.13, standard library only,
no network, deterministic.

Built for [make.fun's Forecast GTA VI's launch-day Twitch viewers hour by hour bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) ($250 in USDC). Referral: https://make.fun/r/wgthwqcw

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Model (see the docstring in `main.py` for detail)

Rolling-midnight console launches only: the forecast is fitted on past
launches whose unlock note says tier "comparable" with a rolling midnight
unlock (no early access), plus unverified console-only launches whose whole
log-ratio curve sits close (RMSE ≤ 0.20) to that core set's median curve.
Two per-hour median estimators are blended in log space (0.7 / 0.3):

1. Theil-Sen regression of log viewers on log hour-1 viewers (robust to
   wild launches);
2. Peak-anchored: OLS of log peak on log hour-1, times the median
   peak-normalised shape.

Each carries its own spread (0.8 × residual/shape std, floored at 0.15 log
units). Before fitting, every comparable's curve is shifted so its detected
US-midnight jump (largest log rise, hours 4–7) lands on the target case's
jump hour — 04:00 UTC (hour 5) in US daylight time, 05:00 UTC (hour 6) on
standard time, from the release date. Guardrails clamp each hour's median
to the subset's observed log-ratio range and floor the day's peak median at
the hour-1 print, so a game bigger than anything in the set still gets a
sane curve. Any failure falls back to flat persistence with a random-walk
spread — the program never crashes and never writes an invalid file.

## Practice validation (local scorer, SullyGnome ground truth)

| case | loss |
|---|---|
| p00 Marvel's Wolverine | 0.1520 |
| p01 Red Dead Redemption 2 | 0.1075 |
| p02 God of War Ragnarök | 0.0900 |
| p03 Marvel's Spider-Man 2 | 0.1953 |
| **overall** | **0.1362** |

House baseline: 0.3871. Practice rank-1 bar: 0.146.
