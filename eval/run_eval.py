#!/usr/bin/env python3
"""
eval/run_eval.py
────────────────
Run analyze_segment() on every row of labelled_segments.csv and report:

  • Precision, Recall, F1  — overall and per-category and per-language
  • All false negatives     (label=1 but not flagged)
  • All false positives     (label=0 but flagged)
  • Saves eval/report.json
  • Exits non-zero if recall < EVAL_RECALL_TARGET (from config)

Usage:
    python eval/run_eval.py [--csv path/to/labelled_segments.csv] [--recall-target 0.80]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import cfg
from pipeline.detector import analyze_segment


def run_eval(csv_path: str, recall_target: float) -> dict:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("[eval] No rows found — check the CSV path.")
        sys.exit(1)

    print(f"[eval] Loaded {len(rows)} rows from {csv_path}")
    print(f"[eval] Flag threshold: {cfg.FLAG_THRESHOLD}")

    tp = fp = fn = tn = 0
    by_category: dict[str, dict[str, int]] = {}
    by_language: dict[str, dict[str, int]] = {}
    false_negatives: list[dict] = []
    false_positives: list[dict] = []

    for row in rows:
        text     = row.get("text", "").strip()
        label    = int(row.get("label", "0"))
        category = row.get("category", "").strip()
        language = row.get("language", "").strip()

        if not text:
            continue

        segment = {"speaker": "EVAL", "text": text, "start": 0.0, "end": 0.0}
        result  = analyze_segment(segment, context_window=[])
        flagged = result["flag"]
        predicted_cat = result.get("category", "")

        # Bucket
        if label == 1 and flagged:
            tp += 1
        elif label == 0 and flagged:
            fp += 1
        elif label == 1 and not flagged:
            fn += 1
            false_negatives.append({
                "text":            text,
                "language":        language,
                "expected_cat":    category,
                "predicted_cat":   predicted_cat,
                "confidence":      result.get("confidence", 0.0),
                "keyword_score":   result.get("keyword_score", 0.0),
                "semantic_score":  result.get("semantic_score", 0.0),
            })
        else:
            tn += 1
            false_positives_target = false_positives  # re-use below for fp

        # FP list
        if label == 0 and flagged:
            false_positives.append({
                "text":           text,
                "language":       language,
                "predicted_cat":  predicted_cat,
                "confidence":     result.get("confidence", 0.0),
            })

        # Per-category buckets
        cat_key = category if category else "none"
        if cat_key not in by_category:
            by_category[cat_key] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        if label == 1 and flagged:
            by_category[cat_key]["tp"] += 1
        elif label == 1 and not flagged:
            by_category[cat_key]["fn"] += 1
        elif label == 0 and flagged:
            by_category.setdefault("none", {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            by_category["none"]["fp"] += 1
        else:
            by_category[cat_key]["tn"] += 1

        # Per-language buckets
        lang_key = language if language else "unknown"
        if lang_key not in by_language:
            by_language[lang_key] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        if label == 1 and flagged:
            by_language[lang_key]["tp"] += 1
        elif label == 1 and not flagged:
            by_language[lang_key]["fn"] += 1
        elif label == 0 and flagged:
            by_language[lang_key]["fp"] += 1
        else:
            by_language[lang_key]["tn"] += 1

    def metrics(tp_, fp_, fn_):
        precision = tp_ / (tp_ + fp_) if (tp_ + fp_) else 0.0
        recall    = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) else 0.0)
        return {"precision": round(precision, 4),
                "recall":    round(recall, 4),
                "f1":        round(f1, 4),
                "tp": tp_, "fp": fp_, "fn": fn_}

    overall = metrics(tp, fp, fn)
    overall["tn"] = tn

    cat_metrics  = {c: metrics(v["tp"], v["fp"], v["fn"]) for c, v in by_category.items()}
    lang_metrics = {l: metrics(v["tp"], v["fp"], v["fn"]) for l, v in by_language.items()}

    # ── Print report ──────────────────────────────────────────────────────
    W = 60
    print(f"\n{'='*W}")
    print("OVERALL")
    print(f"{'='*W}")
    print(f"  Precision : {overall['precision']:.2%}")
    print(f"  Recall    : {overall['recall']:.2%}   (target: {recall_target:.0%})")
    print(f"  F1        : {overall['f1']:.2%}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    print(f"\n{'─'*W}")
    print("PER CATEGORY")
    print(f"{'─'*W}")
    for cat, m in sorted(cat_metrics.items()):
        print(f"  {cat:<18} P={m['precision']:.2%}  R={m['recall']:.2%}  "
              f"F1={m['f1']:.2%}  (TP={m['tp']} FN={m['fn']})")

    print(f"\n{'─'*W}")
    print("PER LANGUAGE")
    print(f"{'─'*W}")
    for lang, m in sorted(lang_metrics.items()):
        print(f"  {lang:<12} P={m['precision']:.2%}  R={m['recall']:.2%}  "
              f"F1={m['f1']:.2%}  (TP={m['tp']} FN={m['fn']})")

    if false_negatives:
        print(f"\n{'─'*W}")
        print(f"FALSE NEGATIVES ({len(false_negatives)})")
        print(f"{'─'*W}")
        for fn_row in false_negatives:
            print(f"  [{fn_row['language']}] {fn_row['text']!r}")
            print(f"    expected={fn_row['expected_cat']}  "
                  f"kw={fn_row['keyword_score']:.2f}  "
                  f"sem={fn_row['semantic_score']:.2f}  "
                  f"conf={fn_row['confidence']:.2f}")

    if false_positives:
        print(f"\n{'─'*W}")
        print(f"FALSE POSITIVES ({len(false_positives)})")
        print(f"{'─'*W}")
        for fp_row in false_positives:
            print(f"  [{fp_row['language']}] {fp_row['text']!r}")
            print(f"    predicted={fp_row['predicted_cat']}  "
                  f"conf={fp_row['confidence']:.2f}")

    report = {
        "overall":         overall,
        "by_category":     cat_metrics,
        "by_language":     lang_metrics,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "recall_target":   recall_target,
        "passed":          overall["recall"] >= recall_target,
    }

    out_path = Path(__file__).parent / "report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[eval] Report saved to {out_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="CallGuard evaluation harness")
    parser.add_argument(
        "--csv", default=str(Path(__file__).parent / "labelled_segments.csv"),
        help="Path to labelled CSV"
    )
    parser.add_argument(
        "--recall-target", type=float, default=cfg.EVAL_RECALL_TARGET,
        help="Minimum recall to pass (default from config)"
    )
    args = parser.parse_args()

    report = run_eval(args.csv, args.recall_target)

    if not report["passed"]:
        print(f"\n[eval] FAILED — recall {report['overall']['recall']:.2%} "
              f"< target {args.recall_target:.0%}")
        sys.exit(1)
    else:
        print(f"\n[eval] PASSED — recall {report['overall']['recall']:.2%} "
              f">= target {args.recall_target:.0%}")


if __name__ == "__main__":
    main()
