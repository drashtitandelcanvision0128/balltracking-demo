"""
Accuracy helpers for cricket ball tracking — pitch calibration, bounce, hit/DOT/RUN.
"""

import math
import cv2
import numpy as np

# Batting crease band (fraction of frame height — ball must be here for hit check)
BATSMAN_ZONE_Y_MIN = 0.22
BATSMAN_ZONE_Y_MAX = 0.92

# Frames to ignore hit detection right after bounce (only skip bounce apex itself)
POST_BOUNCE_HIT_COOLDOWN_FRAMES = 2

# Minimum confidence (0–1) to label a delivery as RUN/HIT
HIT_CONFIDENCE_THRESHOLD = 0.40
POST_BOUNCE_HIT_THRESHOLD = 0.36

# Boundary: post-hit speed vs pre-hit speed ratio
BOUNDARY_SPEED_RATIO = 1.85
BOUNDARY_MIN_LATERAL_PX = 28


def sample_video_frames(cap, width, height, max_samples=40):
    """Grab evenly spaced frames for pitch calibration."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        ret, frame = cap.read()
        return [frame] if ret else []

    indices = np.linspace(int(total * 0.05), int(total * 0.85), max_samples, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frames


def _pitch_mask(frame, height):
    """Mask likely pitch strip (grass / dry turf) in lower portion of frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Tan/brown worn pitch
    tan = cv2.inRange(hsv, (8, 25, 70), (35, 200, 240))
    # Green outfield / pitch
    green = cv2.inRange(hsv, (30, 25, 50), (90, 255, 255))
    mask = cv2.bitwise_or(tan, green)
    mask[: int(height * 0.18), :] = 0
    # Side margins often crowd/stands — focus centre 75%
    side = int(frame.shape[1] * 0.125)
    mask[:, :side] = 0
    mask[:, -side:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def calibrate_pitch_from_video(cap, width, height, fallback_quad, max_samples=40):
    """
    Build a trapezoid homography source quad from pitch colour across sample frames.
    Falls back to auto_pitch_quad when detection is weak.
    """
    frames = sample_video_frames(cap, width, height, max_samples=max_samples)
    if not frames:
        return fallback_quad.copy()

    accum = np.zeros((height, width), dtype=np.float32)
    for frame in frames:
        m = _pitch_mask(frame, height)
        accum += m.astype(np.float32) / 255.0
    accum /= max(len(frames), 1)

    binary = (accum > 0.22).astype(np.uint8) * 255
    if binary.sum() < width * height * 0.02:
        return fallback_quad.copy()

    row_bands = [
        (int(height * 0.52), int(height * 0.58)),
        (int(height * 0.68), int(height * 0.74)),
        (int(height * 0.82), int(height * 0.90)),
    ]
    cx = width * 0.5
    top_pts, bot_pts = [], []

    for y0, y1 in row_bands:
        band = binary[y0:y1, :]
        if band.size == 0:
            continue
        col_sum = band.sum(axis=0).astype(np.float32)
        if col_sum.max() < 1:
            continue
        thresh = col_sum.max() * 0.35
        cols = np.where(col_sum >= thresh)[0]
        if len(cols) < 10:
            continue
        left, right = int(cols[0]), int(cols[-1])
        mid_y = (y0 + y1) // 2
        if mid_y < height * 0.65:
            top_pts.append((left, mid_y, right, mid_y))
        else:
            bot_pts.append((left, mid_y, right, mid_y))

    if not top_pts or not bot_pts:
        return fallback_quad.copy()

    tl_x = int(np.median([p[0] for p in top_pts]))
    tr_x = int(np.median([p[2] for p in top_pts]))
    ty = int(np.median([p[1] for p in top_pts]))
    bl_x = int(np.median([p[0] for p in bot_pts]))
    br_x = int(np.median([p[2] for p in bot_pts]))
    by = int(np.median([p[1] for p in bot_pts]))

    # Sanity: trapezoid wider at bottom (camera behind bowler)
    if (br_x - bl_x) < (tr_x - tl_x) * 0.9:
        return fallback_quad.copy()
    if by <= ty + height * 0.12:
        return fallback_quad.copy()

    quad = np.array([
        [tl_x, ty],
        [tr_x, ty],
        [bl_x, by],
        [br_x, by],
    ], dtype=np.float32)

    # Blend with fallback so extreme detections don't break map
    blend = 0.72
    out = fallback_quad * (1 - blend) + quad * blend
    return out.astype(np.float32)


def refine_bounce_point(raw_pts, height, lookback=14):
    """
    Find bounce apex using local Y minima after downward motion (more stable than 3-frame diff).
    lookback limits search to recent frames so older bounces in the same track are ignored.
    """
    if len(raw_pts) < 5:
        return None
    start = max(0, len(raw_pts) - lookback)
    segment = raw_pts[start:]
    n = len(segment)
    best_i, best_score = None, -1e9
    for i in range(2, n - 2):
        y0, y1, y2, y3, y4 = (segment[j][1] for j in (i - 2, i - 1, i, i + 1, i + 2))
        if not (y1 < y2 and y3 < y2 and y0 < y2):
            continue
        if y2 < height * 0.22:
            continue
        depth = (y2 - y1) + (y2 - y3)
        flatness = abs(y1 - y3)
        score = depth - flatness * 0.5
        if score > best_score:
            best_score = score
            best_i = i
    if best_i is None:
        return None
    return segment[best_i]


def _segment_speeds(raw_pts, fps):
    if len(raw_pts) < 2 or fps <= 0:
        return []
    speeds = []
    for i in range(1, len(raw_pts)):
        d = math.hypot(raw_pts[i][0] - raw_pts[i - 1][0], raw_pts[i][1] - raw_pts[i - 1][1])
        speeds.append(d * fps)
    return speeds


def _angle_between(v1, v2):
    m1, m2 = math.hypot(*v1), math.hypot(*v2)
    if m1 < 1e-3 or m2 < 1e-3:
        return 0.0
    cos_a = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (m1 * m2)))
    return math.degrees(math.acos(cos_a))


def _find_index_near_point(pts, target, max_dist=45):
    best_i, best_d = None, 1e9
    for i, p in enumerate(pts):
        d = math.hypot(p[0] - target[0], p[1] - target[1])
        if d < best_d:
            best_d = d
            best_i = i
    if best_d > max_dist:
        return None
    return best_i


def find_hist_index_near(hist_pts, xy, max_dist=45):
    """Index in smoothed history closest to a video coordinate."""
    return _find_index_near_point(hist_pts, xy, max_dist=max_dist)


def score_hit_post_bounce(hist_pts, height, fps, bounce_idx):
    """
    After pitch bounce, scan forward for bat-contact deflection in batting zone.
    """
    if bounce_idx is None or bounce_idx >= len(hist_pts) - 2:
        return False, 0.0, None

    best_conf = 0.0
    best_contact = None

    # Scan starting from bounce_idx - 1 or bounce_idx to allow hit detection
    # right at the bounce or 1 frame after the bounce.
    # We must ensure we have at least 2 points before and after the candidate contact index.
    start_idx = max(2, bounce_idx - 1)
    end_idx = len(hist_pts) - 2

    for idx in range(start_idx, end_idx):
        p1, p2, p3, p4, p5 = hist_pts[idx - 2], hist_pts[idx - 1], hist_pts[idx], hist_pts[idx + 1], hist_pts[idx + 2]
        cy = p3[1]
        if cy < height * BATSMAN_ZONE_Y_MIN or cy > height * BATSMAN_ZONE_Y_MAX:
            continue

        v_pre = (p3[0] - p1[0], p3[1] - p1[1])
        v_post = (p5[0] - p3[0], p5[1] - p3[1])
        angle = _angle_between(v_pre, v_post)

        pre_speed = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        post_speed = math.hypot(p5[0] - p3[0], p5[1] - p3[1])
        speed_ratio = post_speed / max(pre_speed, 0.5)
        lateral = abs(p5[0] - p3[0])
        forward_pre = p3[1] - p1[1]
        forward_post = p5[1] - p3[1]
        reversed_forward = forward_pre > 1.5 and forward_post < -1.0
        upward_after = (p5[1] < p3[1] - 2) and (p4[1] <= p3[1])

        score = 0.0
        if angle >= 22:
            score += 0.32
        elif angle >= 14:
            score += 0.18
        if speed_ratio >= 1.25:
            score += 0.22
        elif speed_ratio >= 1.08:
            score += 0.10
        if lateral >= 12:
            score += 0.22
        elif lateral >= 6:
            score += 0.10
        if reversed_forward:
            score += 0.18
        if upward_after and angle >= 12:
            score += 0.12
            
        # Prevent false hit on the bounce itself if there's no lateral deflection or reversal
        if abs(idx - bounce_idx) <= 1 and lateral < 8 and not reversed_forward:
            score -= 0.30

        # Bat contact after bounce — don't penalise small bounce-like motion at contact
        rel_idx = idx - bounce_idx
        if rel_idx <= 4 and angle >= 16 and lateral >= 8:
            score += 0.10

        confidence = max(0.0, min(1.0, score))
        if confidence > best_conf:
            best_conf = confidence
            best_contact = p3

    is_hit = best_conf >= POST_BOUNCE_HIT_THRESHOLD
    return is_hit, best_conf, best_contact if is_hit else None


def score_hit(raw_pts, hist_pts, height, fps, bounced, frames_since_bounce, bounce_hist_idx=None):
    """
    Multi-signal hit scorer. Returns (is_hit, confidence 0–1, contact_point or None).
    """
    if len(hist_pts) < 5 or len(raw_pts) < 5:
        return False, 0.0, None

    if bounced and bounce_hist_idx is not None:
        return score_hit_post_bounce(hist_pts, height, fps, bounce_hist_idx)

    p1, p2, p3, p4, p5 = hist_pts[-5], hist_pts[-4], hist_pts[-3], hist_pts[-2], hist_pts[-1]
    contact = p3

    cy = contact[1]
    if cy < height * BATSMAN_ZONE_Y_MIN or cy > height * BATSMAN_ZONE_Y_MAX:
        return False, 0.0, None

    v_pre = (p3[0] - p1[0], p3[1] - p1[1])
    v_post = (p5[0] - p3[0], p5[1] - p3[1])
    angle = _angle_between(v_pre, v_post)

    pre_speed = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    post_speed = math.hypot(p5[0] - p3[0], p5[1] - p3[1])
    speed_ratio = post_speed / max(pre_speed, 0.5)

    # Lateral deflection (hit often sends ball left/right)
    lateral = abs(p5[0] - p3[0])
    # Forward reversal — defensive block or edge
    forward_pre = p3[1] - p1[1]
    forward_post = p5[1] - p3[1]
    reversed_forward = forward_pre > 2 and forward_post < -1.5

    upward_after = (p5[1] < p3[1] - 3) and (p4[1] < p3[1])

    score = 0.0
    if angle >= 28:
        score += 0.35
    elif angle >= 18:
        score += 0.18
    if speed_ratio >= 1.4:
        score += 0.25
    elif speed_ratio >= 1.15:
        score += 0.12
    if lateral >= 18:
        score += 0.20
    elif lateral >= 10:
        score += 0.10
    if reversed_forward:
        score += 0.15
    if upward_after and angle >= 15:
        score += 0.10
    # Full toss hit before bounce — still in batting zone with sharp change
    if not bounced and angle >= 35 and (speed_ratio >= 1.2 or lateral >= 14):
        score += 0.15

    # Penalise bounce-like vertical V at contact (false hit on bounce) — only before bounce registered
    if not bounced and p2[1] < p3[1] and p4[1] < p3[1] and angle < 35:
        score -= 0.25

    # Leave / dot — ball continues straight toward keeper (only when no bounce yet)
    straight_continue = (not bounced) and angle < 12 and lateral < 8 and forward_post > 0
    if straight_continue:
        score -= 0.40

    confidence = max(0.0, min(1.0, score))
    is_hit = confidence >= HIT_CONFIDENCE_THRESHOLD
    return is_hit, confidence, contact if is_hit else None


def classify_boundary(hist_pts, post_hit_max_speed, height, pre_hit_speed):
    """Decide if a hit is a boundary (4/6) vs single run."""
    if len(hist_pts) < 3:
        return False
    p0, p1 = hist_pts[-2], hist_pts[-1]
    step = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    lateral = abs(p1[0] - p0[0])
    high_shot = p1[1] < height * 0.22
    fast = post_hit_max_speed > 38 or (pre_hit_speed > 0 and step / max(pre_hit_speed, 1) >= BOUNDARY_SPEED_RATIO)
    wide = lateral >= BOUNDARY_MIN_LATERAL_PX
    return fast and (wide or high_shot)


def classify_miss(hist_pts, height, hit_occurred, bounced, batsman_y_max=0.92):
    """Ball passed batsman without contact — wicket / leave."""
    if hit_occurred or not bounced or len(hist_pts) < 4:
        return False
    recent = hist_pts[-4:]
    # Monotonic downward in frame (toward keeper/stumps)
    ys = [p[1] for p in recent]
    
    # Dynamic threshold based on batsman_y_max, with a safe upper limit of 70% height
    miss_y_threshold = max(height * 0.70, height * (batsman_y_max - 0.15))
    if ys[-1] < miss_y_threshold:
        return False
        
    downward = all(ys[i + 1] >= ys[i] - 2 for i in range(len(ys) - 1))
    return downward


def snap_to_pitch(px, py, pitch_l, pitch_r, pitch_top, pitch_bot, margin=8):
    px = int(max(pitch_l + margin, min(pitch_r - margin, px)))
    py = int(max(pitch_top + margin, min(pitch_bot - margin, py)))
    return px, py
