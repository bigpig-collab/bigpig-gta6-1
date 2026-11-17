"""GTA VI launch-day Twitch CCU forecast, hours 2..24 -- entry v2.

Same forecaster as v1, with exactly one change. v1 fits every hour on one
nine-game family: console launches with a rolling midnight unlock, no early
access, plus shape-consistent console-only additions. v2 keeps that family
for most hours, but fits hours 10..14 on a narrower console-only family:
the rolling-midnight, tier-comparable launches with no PC release, plus the
same shape gate (RMSE <= 0.20 against the console family's median curve)
that admits Ghost of Tsushima and rejects Saros and Stellar Blade.

Why only midday: launches that also shipped on PC stay flatter through the
middle of the day (the PC audience in Asia/Europe keeps playing), while
console-only launches dip deeper there, and the true midday level of three
of the four practice cases sits closer to the console-only level. The
evening peak and the early climb keep the wider nine-game family, which is
what anchors the level for small hour-1 prints.

Everything else is v1 unchanged: two median estimators per hour blended
0.7/0.3 in log space (Theil-Sen of log viewers on log hour-1, and a
peak-anchored OLS shape), spread 0.8 x residual/shape std floored at 0.15,
US-midnight jump aligned for daylight saving, per-hour guardrail clamps
taken from whichever family fits that hour, a peak floor at the hour-1
print, and a flat persistence fallback when a case cannot be fitted.

Practice (local scorer): loss 0.1306 (p00 0.1410, p01 0.1163, p02 0.0802,
p03 0.1850) vs v1 0.1362. Deterministic, stdlib only, no network.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]

TS_WEIGHT = 0.7          # weight on the Theil-Sen estimator in the blend
SPREAD_MULT = 0.8        # sigma = SPREAD_MULT * residual std
SPREAD_MIN = 0.15        # floor on sigma, in log units
SHAPE_RMSE_LIMIT = 0.20  # shape gate for unverified console-only additions
CLAMP_MARGIN = 0.10      # guardrail margin around a family's observed range
MIDDAY_FIRST = 10        # first hour fitted on the console-only family
MIDDAY_LAST = 14         # last hour fitted on the console-only family


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    half = n // 2
    if n % 2:
        return ordered[half]
    return 0.5 * (ordered[half - 1] + ordered[half])


def _pstdev(values):
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def _ols(xs, ys):
    """Least-squares fit y ~ a + b*x, returning (a, b)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def _theil_sen(xs, ys):
    """Median pairwise slope, intercept as median residual; robust to outliers."""
    slopes = []
    n = len(xs)
    for i in range(n):
        for j in range(i + 1, n):
            if xs[j] != xs[i]:
                slopes.append((ys[j] - ys[i]) / (xs[j] - xs[i]))
    if not slopes:
        return _median(ys), 0.0
    slope = _median(slopes)
    return _median([y - slope * x for x, y in zip(xs, ys)]), slope


def _round2(value):
    """Round to 2 decimals, halves up (same convention as the house starter)."""
    scaled = value * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 100.0
    return int(out) if out == int(out) else out


def _parse_ts(text):
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return None


def _tier(note):
    m = re.search(r"Tier:\s*([A-Za-z\-]+)", note or "")
    return m.group(1).lower() if m else ""


def _unlock_kind(note):
    """Rolling / simultaneous / early / unverified, from the free-text note."""
    text = (note or "").lower()
    if "early access" in text:
        return "early"
    if "rolling" in text or "midnight local" in text:
        return "rolling"
    if "simultaneous" in text or "single global" in text:
        return "simultaneous"
    return "unverified"


def _us_jump_hour(day_start):
    """Hour (1-based) of the US Eastern midnight unlock on the release day.

    04:00 UTC (hour 5) under US daylight time, 05:00 UTC (hour 6) under
    standard time. Daylight time runs from the 2nd Sunday of March to the
    1st Sunday of November. None when the date cannot be parsed.
    """
    day = _parse_ts(day_start or "")
    if day is None:
        return None

    def nth_sunday(year, month, n):
        first = datetime(year, month, 1)
        return first + timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))

    dst_start = nth_sunday(day.year, 3, 2).date()
    dst_end = nth_sunday(day.year, 11, 1).date()
    return 5 if dst_start <= day.date() < dst_end else 6


def _jump_hour(series):
    """Hour in 4..7 with the sharpest log rise: the comparable's US unlock."""
    best_hour, best_rise = 5, None
    for h in range(4, 8):
        if h in series and h - 1 in series and series[h - 1] > 0:
            rise = math.log(series[h] / series[h - 1])
            if best_rise is None or rise > best_rise:
                best_rise, best_hour = rise, h
    return best_hour


def _align(family, target_jump):
    """Shift each comparable so its detected jump lands on target_jump.

    Hour 1 is the anchor and never moves; an hour that would fall off the
    1..24 edge keeps its own value instead of disappearing.
    """
    if target_jump is None:
        return family
    aligned = []
    for comp in family:
        shift = target_jump - _jump_hour(comp["series"])
        series = {}
        for h in range(1, 25):
            src = h - shift
            if 1 <= src <= 24 and src in comp["series"]:
                series[h] = comp["series"][src]
            elif h in comp["series"]:
                series[h] = comp["series"][h]
        series[1] = comp["p1"]
        aligned.append({"p1": comp["p1"], "series": series})
    return aligned


def _day_series(comp):
    """Hourly viewers (hours 1..24) on the comparable's release day."""
    day0 = _parse_ts(comp.get("day_start", ""))
    if day0 is None:
        return {}
    out = {}
    for row in comp.get("hourly") or []:
        when = _parse_ts(row.get("from", ""))
        viewers = row.get("viewers")
        if when is None or not isinstance(viewers, (int, float)) or viewers <= 0:
            continue
        h = int((when - day0).total_seconds() // 3600) + 1
        if 1 <= h <= 24:
            out[h] = float(viewers)
    return out


def _median_log_ratio(family, hour):
    vals = [math.log(c["series"][hour] / c["p1"])
            for c in family if hour in c["series"]]
    return _median(vals) if vals else None


def _build_family(comparables, console_only_core):
    """The launch family a fit is taken from.

    Core: tier "comparable" + rolling midnight unlock. With
    console_only_core the core is additionally restricted to launches with
    no PC release. On top of the core, an unverified console-only
    comparable joins when its whole log-ratio curve is close (RMSE <= 0.20)
    to the core's median curve.
    """
    parsed = []
    for comp in comparables:
        series = _day_series(comp)
        if 1 not in series or len(series) < 18:
            continue
        parsed.append({
            "p1": series[1],
            "series": series,
            "tier": _tier(comp.get("unlock", "")),
            "kind": _unlock_kind(comp.get("unlock", "")),
            "console_only": "PC" not in (comp.get("platforms") or ""),
        })
    core = [c for c in parsed
            if c["tier"] == "comparable" and c["kind"] == "rolling"
            and (c["console_only"] or not console_only_core)]
    if len(core) < 4:
        core = [c for c in parsed
                if c["tier"] == "comparable" and c["kind"] != "early"
                and (c["console_only"] or not console_only_core)]
    if len(core) < 3:
        core = [c for c in parsed
                if c["tier"] not in ("early-access-contaminated", "weak",
                                     "target-history")
                and (c["console_only"] or not console_only_core)]
    if len(core) < 3:
        return [c for c in parsed
                if c["console_only"] or not console_only_core]

    family = list(core)
    # Median log-ratio curve of the core, computed once and reused by the
    # shape gate for every unverified candidate.
    core_curve = {}
    for h in range(2, 25):
        med = _median_log_ratio(core, h)
        if med is not None:
            core_curve[h] = med
    for c in parsed:
        if c in core or c in family:
            continue
        if c["tier"] != "comparable" or c["kind"] != "unverified":
            continue
        if not c["console_only"]:
            continue
        ratios = {h: math.log(v / c["p1"])
                  for h, v in c["series"].items() if h >= 2}
        common = [h for h in ratios if h in core_curve]
        if len(common) < 12:
            continue
        rmse = math.sqrt(sum((ratios[h] - core_curve[h]) ** 2
                             for h in common) / len(common))
        if rmse <= SHAPE_RMSE_LIMIT:
            family.append(c)
    return family


def _fit_theil_sen(family, p1):
    """Per hour: Theil-Sen log(viewers_h) ~ log(p1). Returns {h: (log_med, sigma)}."""
    log_p1 = math.log(p1)
    out = {}
    for h in range(2, 25):
        xs, ys = [], []
        for c in family:
            if h in c["series"]:
                xs.append(math.log(c["p1"]))
                ys.append(math.log(c["series"][h]))
        if len(xs) < 3:
            continue
        a, b = _theil_sen(xs, ys)
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        out[h] = (a + b * log_p1, max(SPREAD_MIN, SPREAD_MULT * _pstdev(resid)))
    return out


def _fit_peak_anchored(family, p1):
    """Per hour: OLS log(peak) ~ log(p1), times the median peak-normalised shape."""
    log_p1 = math.log(p1)
    xs, peaks = [], []
    for c in family:
        xs.append(math.log(c["p1"]))
        peaks.append(max(c["series"].values()))
    if len(xs) < 3:
        return {}
    a, b = _ols(xs, [math.log(p) for p in peaks])
    log_peak = a + b * log_p1
    out = {}
    for h in range(2, 25):
        shapes = [math.log(c["series"][h] / max(c["series"].values()))
                  for c in family if h in c["series"]]
        if not shapes:
            continue
        out[h] = (log_peak + _median(shapes),
                  max(SPREAD_MIN, SPREAD_MULT * _pstdev(shapes)))
    return out


def _persistence(p1):
    """Flat fallback: every hour at hour 1, with a random-walk spread."""
    return {h: [_round2(p1 * math.exp(z * 0.15 * math.sqrt(h - 1)))
                for z in Z_SCORES] for h in range(2, 25)}


def _forecast(case):
    """Return {hour: [q05, q25, q50, q75, q95]} for hours 2..24."""
    try:
        known = case.get("known") or []
        first = next((k for k in known if k.get("hour") == 1), None)
        p1 = float(first["viewers"]) if first and first.get("viewers") else 1.0
    except (TypeError, ValueError, KeyError):
        p1 = 1.0
    if not (p1 > 0) or not math.isfinite(p1):
        p1 = 1.0

    try:
        wide = _build_family(case.get("comparables") or [], False)
        narrow = _build_family(case.get("comparables") or [], True)
        target = _us_jump_hour(case.get("day_start", ""))
        if target is None:
            jumps = [_jump_hour(c["series"]) for c in wide]
            target = _median(jumps) if jumps else 5
        wide = _align(wide, target)
        narrow = _align(narrow, target)
        fits = {}
        for h in range(2, 25):
            family = narrow if MIDDAY_FIRST <= h <= MIDDAY_LAST else wide
            ts_fit = _fit_theil_sen(family, p1)
            pa_fit = _fit_peak_anchored(family, p1)
            if h in ts_fit and h in pa_fit:
                fits[h] = (family, ts_fit[h], pa_fit[h])
        if len(fits) < 12:
            raise ValueError("too few fittable hours")
    except Exception:
        return _persistence(p1)

    # Blend the two estimators' quantiles in log space, hour by hour.
    rows = {}
    for h, (family, (m1, s1), (m2, s2)) in fits.items():
        rows[h] = [math.exp(TS_WEIGHT * (m1 + z * s1)
                            + (1.0 - TS_WEIGHT) * (m2 + z * s2))
                   for z in Z_SCORES]
    if len(rows) < 12:
        return _persistence(p1)

    # Guardrails, from the fitting family's own observed log-ratio ranges.
    med_log = {}
    for h, qs in rows.items():
        family = fits[h][0]
        vals = [math.log(c["series"][h] / c["p1"])
                for c in family if h in c["series"]]
        lr = math.log(qs[2] / p1)
        if vals:
            lo = min(vals) - CLAMP_MARGIN
            hi = max(vals) + CLAMP_MARGIN
            lr = max(lo, min(hi, lr))
        med_log[h] = lr
    peak_ratio = max(math.exp(v) for v in med_log.values())
    lift = math.log(1.0 / peak_ratio) if peak_ratio < 1.0 else 0.0
    for h, qs in rows.items():
        shift = med_log[h] + (lift if h >= 6 else 0.0) - math.log(qs[2] / p1)
        rows[h] = [q * math.exp(shift) for q in qs]

    # Finalise: round, keep in (0, 1e9], enforce non-decreasing quantiles.
    final = {}
    for h in range(2, 25):
        qs = rows.get(h)
        if qs is None:
            qs = [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z_SCORES]
        out = []
        prev = 0.0
        for q in qs:
            v = _round2(q)
            if not math.isfinite(v) or v <= 0:
                v = _round2(prev) if prev > 0 else 0.01
            v = min(v, 1e9)
            v = max(v, prev)
            out.append(v)
            prev = v
        final[h] = out
    return final


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            rows = _forecast(case)
        except Exception:
            rows = _persistence(1.0)
        payload = {"hours": [{"hour": h, "q": rows[h]} for h in range(2, 25)]}
        text = json.dumps(payload, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
