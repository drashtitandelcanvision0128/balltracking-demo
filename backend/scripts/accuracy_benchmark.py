"""
Accuracy benchmark — compare tracked deliveries against ground-truth JSON.

Usage:
  python scripts/accuracy_benchmark.py --video match.mp4 --ground-truth benchmarks/deliveries.json
  python scripts/accuracy_benchmark.py --results job_result.json --ground-truth benchmarks/deliveries.json

Ground-truth format (deliveries.json):
{
  "video": "match.mp4",
  "fps": 30,
  "deliveries": [
    {
      "frame": 120,
      "bounce_x_m": 0.12,
      "bounce_y_m": 5.4,
      "speed_kmh": 128,
      "outcome": "DOTS",
      "length_type": "GOOD LENGTH"
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.classifier import classify_length, classify_line  # noqa: E402


def _match_deliveries(predicted: list[dict], ground: list[dict], frame_tol: int = 45) -> list[tuple[dict, dict]]:
    pairs = []
    used = set()
    for gt in ground:
        gf = int(gt.get("frame", 0))
        best, best_d = None, frame_tol + 1
        for i, pred in enumerate(predicted):
            if i in used:
                continue
            pf = int(pred.get("frame", 0))
            d = abs(pf - gf)
            if d <= frame_tol and d < best_d:
                best, best_d = i, d
        if best is not None:
            used.add(best)
            pairs.append((predicted[best], gt))
    return pairs


def evaluate(predicted: list[dict], ground_truth: dict) -> dict:
    gt_deliveries = ground_truth.get("deliveries", [])
    pairs = _match_deliveries(predicted, gt_deliveries)

    if not gt_deliveries:
        return {"error": "No ground-truth deliveries"}

    bounce_errors = []
    speed_errors = []
    length_ok = 0
    outcome_ok = 0
    detected_frames = 0
    total_gt_frames = sum(int(d.get("track_frames", 30)) for d in gt_deliveries)

    for pred, gt in pairs:
        bx = pred.get("bounce_x")
        by = pred.get("bounce_y")
        if bx is not None and by is not None:
            ex = float(gt.get("bounce_x_m", bx)) - float(bx)
            ey = float(gt.get("bounce_y_m", by)) - float(by)
            bounce_errors.append(math.hypot(ex, ey))

        ps = float(pred.get("speed_kmh", 0))
        gs = float(gt.get("speed_kmh", 0))
        if ps > 0 and gs > 0:
            speed_errors.append(abs(ps - gs))

        pl = pred.get("length_type") or pred.get("length", "")
        gl = gt.get("length_type", "")
        if pl and gl and pl.upper() == gl.upper():
            length_ok += 1

        po = str(pred.get("type", pred.get("outcome", ""))).upper()
        go = str(gt.get("outcome", "")).upper()
        if po and go and po == go:
            outcome_ok += 1

        detected_frames += int(pred.get("track_frames", 25))

    n = len(pairs)
    n_gt = len(gt_deliveries)
    return {
        "matched_deliveries": n,
        "ground_truth_deliveries": n_gt,
        "match_rate_pct": round(100 * n / max(n_gt, 1), 1),
        "bounce_position_error_m": {
            "mean": round(sum(bounce_errors) / len(bounce_errors), 3) if bounce_errors else None,
            "max": round(max(bounce_errors), 3) if bounce_errors else None,
            "target": 0.5,
            "pass": (sum(bounce_errors) / len(bounce_errors) < 0.5) if bounce_errors else False,
        },
        "speed_error_kmh": {
            "mean": round(sum(speed_errors) / len(speed_errors), 2) if speed_errors else None,
            "max": round(max(speed_errors), 2) if speed_errors else None,
            "target": 5.0,
            "pass": (sum(speed_errors) / len(speed_errors) < 5.0) if speed_errors else False,
        },
        "length_zone_accuracy_pct": round(100 * length_ok / max(n, 1), 1),
        "length_target_pct": 80,
        "outcome_accuracy_pct": round(100 * outcome_ok / max(n, 1), 1),
        "outcome_target_pct": 75,
        "detection_rate_pct": round(100 * detected_frames / max(total_gt_frames, 1), 1),
        "detection_target_pct": 85,
    }


def main():
    parser = argparse.ArgumentParser(description="Cricket tracking accuracy benchmark")
    parser.add_argument("--ground-truth", required=True, help="Path to ground-truth JSON")
    parser.add_argument("--results", help="Job result JSON with bounce_events")
    parser.add_argument("--video", help="Process video and benchmark (requires backend)")
    parser.add_argument("--manual-quad", help="JSON array of 4 [x,y] pitch corners")
    parser.add_argument("--output", help="Write report JSON here")
    args = parser.parse_args()

    with open(args.ground_truth, encoding="utf-8") as f:
        gt = json.load(f)

    if args.results:
        with open(args.results, encoding="utf-8") as f:
            data = json.load(f)
        predicted = data.get("bounce_events") or data.get("bounces") or []
    elif args.video:
        import predict_server as ps

        options = {}
        if args.manual_quad:
            options["manual_quad"] = json.loads(args.manual_quad)
        out = os.path.join(ps.UPLOAD_FOLDER, f"bench_{os.path.basename(args.video)}")
        result = ps.process_video(args.video, out, options=options)
        predicted = result.get("bounce_events", [])
    else:
        parser.error("Provide --results or --video")

    report = evaluate(predicted, gt)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
