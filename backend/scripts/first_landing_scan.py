"""Count deliveries and mark first floor landing on a rear-view indoor clip."""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def _pink_yellow_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pink1 = cv2.inRange(hsv, (0, 70, 90), (12, 255, 255))
    pink2 = cv2.inRange(hsv, (150, 70, 90), (179, 255, 255))
    yellow = cv2.inRange(hsv, (18, 80, 90), (42, 255, 255))
    mask = cv2.bitwise_or(cv2.bitwise_or(pink1, pink2), yellow)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask


def _blobs(mask: np.ndarray, height: int, width: int, motion: np.ndarray | None = None) -> list[tuple[float, float, float, float]]:
    work = mask if motion is None else cv2.bitwise_and(mask, motion)
    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = float(w * h)
        if area < 22 or area > 3200:
            continue
        if w < 5 or h < 5:
            continue
        ratio = max(w, h) / max(min(w, h), 1)
        if ratio > 2.8:
            continue
        cx = x + w * 0.5
        yb = float(y + h)
        if yb < height * 0.30:
            continue
        out.append((cx, yb, area, float(max(w, h))))
    out.sort(key=lambda p: -p[2])
    return out[:6]


def _split_deliveries(pts: list, fps: float) -> list[list]:
    if not pts:
        return []
    gap = max(18, int(fps * 0.55))
    groups, cur = [], [pts[0]]
    for p in pts[1:]:
        if p[0] - cur[-1][0] > gap:
            groups.append(cur)
            cur = [p]
        else:
            cur.append(p)
    groups.append(cur)
    # keep tracks that actually travel toward the batsman (y drops / area shrinks)
    kept = []
    for g in groups:
        if len(g) < max(8, int(fps * 0.18)):
            continue
        y0, y1 = g[0][2], g[-1][2]
        a0 = max(g[i][3] for i in range(min(5, len(g))))
        a1 = min(g[i][3] for i in range(max(0, len(g) - 5), len(g)))
        travel = abs(g[-1][1] - g[0][1]) + abs(y1 - y0)
        toward_batsman = (y0 - y1) > 24 or (a0 > a1 * 1.25)
        if travel < 36 or not toward_batsman:
            continue  # static / leftover ball
        kept.append(g)
    return kept


def _first_landing(g: list, height: int, width: int, fps: float) -> tuple[int, float, float] | None:
    from core.pipeline.first_pitch_point import find_bounce_reversal_point
    pts = [(int(p[0]), float(p[1]), float(p[2]), float(p[3])) for p in g]
    hit = find_bounce_reversal_point(
        pts, int(pts[0][0]), height=height, width=width, fps=fps,
    )
    if hit is None:
        return None
    f, x, y, _ = hit
    return int(f), float(x), float(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    video = args.video
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    pts = []
    last = None
    prev_gray = None
    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fidx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = None
        if prev_gray is not None:
            delta = cv2.absdiff(gray, prev_gray)
            _, motion = cv2.threshold(delta, 16, 255, cv2.THRESH_BINARY)
            motion = cv2.dilate(motion, None, iterations=2)
        prev_gray = gray
        blobs = _blobs(_pink_yellow_mask(frame), height, width, motion=motion)
        chosen = None
        if blobs:
            if last is None:
                chosen = max(blobs, key=lambda b: b[2] + b[1] * 0.12)
            else:
                lx, ly = last
                scored = []
                for b in blobs:
                    dist = ((b[0] - lx) ** 2 + (b[1] - ly) ** 2) ** 0.5
                    if dist > 160:
                        continue
                    scored.append((dist, b))
                chosen = min(scored, key=lambda t: t[0])[1] if scored else max(blobs, key=lambda b: b[2])
        if chosen is not None:
            pts.append((fidx, float(chosen[0]), float(chosen[1]), float(chosen[2])))
            last = (chosen[0], chosen[1])
        else:
            last = None
        if fidx % 400 == 0:
            print(f"scanned {fidx}/{total} moving_hits={len(pts)}", flush=True)
    cap.release()

    groups = _split_deliveries(pts, fps)
    landings = []
    for i, g in enumerate(groups, 1):
        hit = _first_landing(g, height, width, fps)
        landings.append({"delivery": i, "frames": len(g), "start": g[0][0], "end": g[-1][0], "landing": hit})
        print(
            f"delivery {i}: frames {g[0][0]}-{g[-1][0]} "
            f"({g[0][0]/fps:.1f}-{g[-1][0]/fps:.1f}s) n={len(g)} landing={hit}",
            flush=True,
        )

    out = args.out or os.path.join(
        os.path.dirname(video), "first_landing_" + os.path.splitext(os.path.basename(video))[0] + ".mp4"
    )
    cap = cv2.VideoCapture(video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out, fourcc, fps, (width, height))
    marks = [(d["landing"], d["delivery"]) for d in landings if d["landing"]]
    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fidx += 1
        for (lf, x, y), did in marks:
            if fidx >= lf:
                cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (int(x), int(y)), 10, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(
                    frame, str(did), (int(x) + 12, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
                )
        cv2.putText(
            frame, f"Balls to batsman: {len(marks)}", (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
        )
        writer.write(frame)
    cap.release()
    writer.release()
    summary = os.path.splitext(out)[0] + "_summary.jpg"
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
    ok, still = cap.read()
    cap.release()
    if ok:
        for (lf, x, y), did in marks:
            cv2.circle(still, (int(x), int(y)), 11, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(still, (int(x), int(y)), 11, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                still, f"#{did}", (int(x) + 12, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
            )
        cv2.putText(
            still, f"{len(marks)} balls  |  red = first landing", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA,
        )
        cv2.imwrite(summary, still)
    meta = os.path.splitext(out)[0] + ".json"
    with open(meta, "w", encoding="utf-8") as fh:
        json.dump({"video": video, "deliveries": len(marks), "landings": landings}, fh, indent=2)
    print(f"DONE deliveries={len(marks)} video={out} summary={summary}", flush=True)


if __name__ == "__main__":
    main()
