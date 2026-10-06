#!/usr/bin/env python3
"""Rank a campaign's points and say what the comparison actually supports.

    run/campaign-summary.py <sweep dir> <out root>

Reads <sweep>/<engine>-<shape>/metrics.csv, the file observe.sh writes, one row
per concurrency point.

Two things this does that a plain sort does not:

  * It refuses to name a winner across a difference smaller than a short
    measurement's own spread. A 3% gap between two engines at a 20-second window
    is not a result, and printing it as one is how a write-up ends up defending a
    number that will not reproduce.

  * It reports the solo/PD comparison per engine rather than pooling engines,
    because the two questions -- "does disaggregation help" and "which engine is
    faster" -- have different answers and pooling them hides both.

The scored metric is good output tokens/sec = goodput x OSL, and a point only
counts at all when good_request_fraction >= 0.90. Points below the gate are
printed but never ranked: they are not slow, they are unscoreable.
"""
import csv
import os
import sys

GATE = 0.90
# Below this, two configurations are not distinguishable at these window lengths.
# Not a statistical claim -- an honesty threshold. Job pairs 18964/18965 on Firmus
# put repeatability at a few percent; until this campaign measures its own, treat
# anything under 5% as a tie and say so.
TIE_PCT = 5.0


def num(row, key, default=None):
    try:
        return float(row.get(key) or "")
    except (TypeError, ValueError):
        return default


def load(sweep):
    rows = []
    for cell in sorted(os.listdir(sweep)):
        path = os.path.join(sweep, cell, "metrics.csv")
        if not os.path.isfile(path):
            continue
        # cell is engine-shape or engine-shape+variant; '+' separates the variant
        # so a shape like 4x1_4x1 stays intact.
        engine, _, rest = cell.partition("-")
        shape, _, variant = rest.partition("+")
        shape = shape.replace("_", ":")
        variant = variant or "base"
        with open(path) as fh:
            for r in csv.DictReader(fh):
                goodput = num(r, "goodput")
                if goodput is None:
                    continue
                osl = num(r, "osl") or 200.0
                rows.append({
                    "engine": engine,
                    "shape": shape,
                    "variant": variant,
                    "point": r.get("point", "?"),
                    "n": int(str(r.get("point", "c0")).lstrip("c") or 0),
                    "goodput": goodput,
                    "toks": goodput * osl,
                    "pass": num(r, "pass_rate", 0.0) or 0.0,
                    "ttft": num(r, "ttft_p90"),
                    "itl": num(r, "itl_p90") or num(r, "itl_avg"),
                })
    return rows


def table(rows, gpus):
    out = [" %-8s %-8s %-11s %5s %11s %9s %7s %9s %8s %6s" % (
        "engine", "shape", "variant", "N", "good tok/s", "/GPU", "pass",
        "ttft p90", "itl p90", "gate")]
    out.append(" " + "-" * 95)
    for r in rows:
        out.append(" %-8s %-8s %-11s %5d %11.0f %9.1f %7.3f %9.0f %8.1f %6s" % (
            r["engine"], r["shape"], r["variant"], r["n"], r["toks"], r["toks"] / gpus,
            r["pass"], r["ttft"] or 0, r["itl"] or 0,
            "pass" if r["pass"] >= GATE else "FAIL"))
    return out


def main():
    sweep, root = sys.argv[1], sys.argv[2]
    gpus = float(os.environ.get("CAMPAIGN_GPUS", "8"))
    label = os.environ.get("GPU_LABEL", os.environ.get("SITE_NAME", "unknown machine"))

    window = os.environ.get("BENCH_SECONDS", "?")
    warmup = os.environ.get("WARMUP_SECONDS", "?")
    official = (window == "120" and warmup == "60")

    rows = load(sweep)
    out = ["", "=" * 84,
           " campaign on %s -- %g GPUs" % (label, gpus),
           " window: warmup %ss / benchmark %ss%s" % (
               warmup, window,
               "  <- the official durations" if official
               else "  (exploratory; the scored run is warmup 60s / benchmark 120s)"),
           " scored metric: good output tok/s = goodput x OSL, only where"
           " good_request_fraction >= %.2f" % GATE,
           "=" * 84]

    if not rows:
        out.append("")
        out.append(" no parsable results -- every cell failed to produce metrics.csv.")
        out.append(" Look at sweep/<cell>/deploy/*.log first: a cell that never came up")
        out.append(" writes no metrics at all, which is a deployment failure, not a slow run.")
        print("\n".join(out))
        return 1

    scoreable = sorted([r for r in rows if r["pass"] >= GATE], key=lambda r: -r["toks"])
    failing = sorted([r for r in rows if r["pass"] < GATE], key=lambda r: (r["engine"], r["shape"], r["n"]))

    out.append("")
    out.append(" RANKED -- points clearing the gate")
    if scoreable:
        out += table(scoreable, gpus)
    else:
        out.append("   none. Every point missed good_request_fraction >= %.2f, so nothing here" % GATE)
        out.append("   would score. Drop the concurrencies and measure again.")

    if failing:
        out.append("")
        out.append(" BELOW THE GATE -- unscoreable, listed for the slope only")
        out += table(failing, gpus)

    # Best per engine, and the solo/PD question per engine.
    out.append("")
    out.append(" BEST PER ENGINE")
    by_engine = {}
    for r in scoreable:
        by_engine.setdefault(r["engine"], []).append(r)
    for engine in sorted(set(r["engine"] for r in rows)):
        best = max(by_engine.get(engine, []), key=lambda r: r["toks"], default=None)
        if best is None:
            out.append("   %-8s nothing cleared the gate" % engine)
            continue
        out.append("   %-8s %-9s N=%-3d %8.0f good tok/s  (%.1f /GPU, itl p90 %.1f ms)" % (
            engine, best["shape"], best["n"], best["toks"], best["toks"] / gpus, best["itl"] or 0))

    out.append("")
    out.append(" DISAGGREGATION, PER ENGINE (the only comparison that isolates it)")
    for engine in sorted(set(r["engine"] for r in rows)):
        mine = by_engine.get(engine, [])
        solo = max([r for r in mine if r["shape"].startswith("solo")], key=lambda r: r["toks"], default=None)
        pd = max([r for r in mine if not r["shape"].startswith("solo")], key=lambda r: r["toks"], default=None)
        if not solo or not pd:
            have = ", ".join(sorted(set(r["shape"] for r in mine))) or "nothing"
            out.append("   %-8s cannot compare -- cleared the gate only for: %s" % (engine, have))
            continue
        delta = 100.0 * (pd["toks"] - solo["toks"]) / solo["toks"]
        verdict = "PD ahead" if delta > TIE_PCT else ("solo ahead" if delta < -TIE_PCT else "tie")
        out.append("   %-8s solo %.0f (N=%d) vs PD %.0f (N=%d) -> %+.1f%%  %s" % (
            engine, solo["toks"], solo["n"], pd["toks"], pd["n"], delta, verdict))
    out.append("   a gap under %.0f%% is inside what a %ss window can produce by itself." % (
        TIE_PCT, os.environ.get("BENCH_SECONDS", "20")))

    # Variants, against their own baseline. Comparing a variant to the global best
    # would mix in the engine and the shape; comparing it to `base` of the same
    # engine and shape isolates the one thing it changed.
    variants = sorted({r["variant"] for r in rows})
    if len(variants) > 1:
        out.append("")
        out.append(" VARIANTS, each against `base` of the same engine and shape")
        for engine in sorted({r["engine"] for r in rows}):
            for shape in sorted({r["shape"] for r in rows if r["engine"] == engine}):
                def best_of(v):
                    c = [r for r in scoreable
                         if r["engine"] == engine and r["shape"] == shape and r["variant"] == v]
                    return max(c, key=lambda r: r["toks"], default=None)
                b = best_of("base")
                for v in variants:
                    if v == "base":
                        continue
                    r = best_of(v)
                    if r is None:
                        out.append("   %-8s %-8s %-11s nothing cleared the gate" % (engine, shape, v))
                        continue
                    if b is None:
                        out.append("   %-8s %-8s %-11s %8.0f (N=%d) -- no base to compare with" % (
                            engine, shape, v, r["toks"], r["n"]))
                        continue
                    d = 100.0 * (r["toks"] - b["toks"]) / b["toks"]
                    verdict = "better" if d > TIE_PCT else ("worse" if d < -TIE_PCT else "tie")
                    out.append("   %-8s %-8s %-11s %8.0f (N=%-3d) vs base %8.0f (N=%-3d)  %+6.1f%%  %s" % (
                        engine, shape, v, r["toks"], r["n"], b["toks"], b["n"], d, verdict))
        out.append("   Anything inside %.0f%% is a tie at these window lengths." % TIE_PCT)

    # The winner, stated only as strongly as the margin allows.
    if scoreable:
        best = scoreable[0]
        others = [r for r in scoreable
                  if (r["engine"], r["shape"], r["variant"])
                  != (best["engine"], best["shape"], best["variant"])]
        out.append("")
        out.append(" TOP POINT  %s %s %s at N=%d -- %.0f good output tok/s (%.1f per GPU)" % (
            best["engine"], best["shape"], best["variant"], best["n"], best["toks"],
            best["toks"] / gpus))
        if others:
            runner = max(others, key=lambda r: r["toks"])
            margin = 100.0 * (best["toks"] - runner["toks"]) / best["toks"]
            out.append(" runner-up  %s %s %s at N=%d -- %.0f  (margin %.1f%%)" % (
                runner["engine"], runner["shape"], runner["variant"], runner["n"],
                runner["toks"], margin))
            if margin < TIE_PCT:
                out.append(" ! that margin is inside the noise of a short window. These two are")
                out.append("   tied until both are re-measured at the official 120 seconds.")

        # Where the gate bites. This is the whole reason a high-throughput
        # configuration can score zero: ITL, not tokens/sec, is the constraint.
        near = [r for r in scoreable if (r["itl"] or 0) > 0.85 * 20.0]
        if near:
            out.append("")
            out.append(" ! %d scoring point(s) sit within 15%% of the 20 ms ITL gate:" % len(near))
            for r in near[:6]:
                out.append("     %s %s N=%d  itl p90 %.1f ms" % (
                    r["engine"], r["shape"], r["n"], r["itl"]))
            out.append("   The next concurrency step up probably scores nothing at all.")

    # A markdown block, so the results log and the dashboard get the same numbers
    # as this printout rather than a hand-copied version of them.
    md = ["", " MARKDOWN (for docs/results-log.md and the dashboard)", ""]
    md.append("| engine | shape | variant | machine | GPU | N | window | good tok/s | /GPU | TTFT p90 | ITL p90 | gate |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    win = "**%ss official**" % window if official else "%ss" % window
    for r in scoreable + failing:
        md.append("| %s | %s | %s | %s | %g | %d | %s | %.0f | %.1f | %.0f | %.1f | %s |" % (
            r["engine"], r["shape"], r["variant"], label, gpus, r["n"], win, r["toks"],
            r["toks"] / gpus, r["ttft"] or 0, r["itl"] or 0,
            "pass %.2f" % r["pass"] if r["pass"] >= GATE else "**fail %.2f**" % r["pass"]))
    out += md

    text = "\n".join(out) + "\n"
    print(text)
    with open(os.path.join(root, "campaign-summary.txt"), "w") as fh:
        fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
