"""
Hawk-Eye / broadcast-style cricket pitch map renderer.
Full length zones: Full Toss, Yorker, Half Volley, Full, Length, Back of Length, Short.
"""

import cv2
import numpy as np

MAP_W, MAP_H = 720, 960
PITCH_L, PITCH_R = 185, 535
PITCH_TOP, PITCH_BOT = 70, 780
CENTER_X = (PITCH_L + PITCH_R) // 2
LABEL_X = PITCH_R + 8

# Zone boundaries (y from top — stumps end)
Y_YORKER = 130
Y_2M, Y_4M, Y_6M, Y_8M = 195, 295, 395, 495
Y_HALFWAY = 600

# (y1, y2, color BGR, light BGR, label)
PITCH_ZONES = [
    (PITCH_TOP,  Y_YORKER,   (200,  60, 180), (220, 180, 240), 'FULL TOSS'),
    (Y_YORKER,   Y_2M,       (0,  220, 255), (210, 240, 255), 'YORKER'),
    (Y_2M,       Y_4M,       (255, 200, 120), (230, 235, 255), 'HALF VOLLEY'),
    (Y_4M,       Y_6M,       (80,  200,  80), (210, 255, 210), 'FULL'),
    (Y_6M,       Y_8M,       (50,  140, 255), (210, 220, 255), 'LENGTH'),
    (Y_8M,       Y_HALFWAY,  (50,   50, 220), (210, 210, 255), 'BACK OF A LENGTH'),
    (Y_HALFWAY,  PITCH_BOT,  (140, 140, 140), (225, 225, 225), 'SHORT'),
]

ZONE_NAMES = [z[4] for z in PITCH_ZONES]

OUTCOME_COLORS_BGR = {
    'DOTS': (20, 20, 20),
    'RUNS': (255, 210, 0),
    'BOUNDARIES': (220, 120, 30),
    'WICKETS': (255, 255, 255),
}

LIGHT_OUTCOME_BGR = {
    'DOTS': (60, 200, 60),       # green — reference style
    'RUNS': (60, 60, 220),       # red
    'BOUNDARIES': (40, 140, 255), # orange
    'WICKETS': (200, 200, 200),
}

RUNS_LEGEND = [
    ('0', (20, 20, 20)),
    ('1', (0, 220, 255)),
    ('2', (255, 200, 120)),
    ('3', (80, 200, 80)),
    ('4', (220, 120, 30)),
    ('5', (50, 140, 255)),
    ('6', (50, 50, 220)),
    ('Out', (255, 255, 255)),
]

TEMPLATE_CORNERS = np.array([
    [PITCH_L, PITCH_TOP],
    [PITCH_R, PITCH_TOP],
    [PITCH_L, PITCH_BOT],
    [PITCH_R, PITCH_BOT],
], dtype=np.float32)


def classify_length_zone(py):
    """Return length label from pitch-map y coordinate."""
    py = int(py)
    for y1, y2, _, _, label in PITCH_ZONES:
        if y1 <= py < y2:
            return label
    return 'SHORT'


def zone_stats(bounces):
    """Percentage of balls in each length zone."""
    counts = {name: 0 for name in ZONE_NAMES}
    for b in bounces:
        cy = b['coords'][1] if isinstance(b['coords'], (list, tuple)) else b['coords'][1]
        counts[classify_length_zone(cy)] += 1
    total = max(len(bounces), 1)
    return {name: int(round(100 * counts[name] / total)) for name in ZONE_NAMES}


def _draw_pitch_zones(img, light=False):
    for y1, y2, col, light_col, _ in PITCH_ZONES:
        c = light_col if light else col
        cv2.rectangle(img, (PITCH_L, y1), (PITCH_R, y2), c, -1)
    cv2.rectangle(img, (PITCH_L, PITCH_TOP), (PITCH_R, PITCH_BOT), (255, 255, 255), 2)
    cv2.line(img, (CENTER_X, PITCH_TOP), (CENTER_X, PITCH_BOT), (200, 120, 60), 2, cv2.LINE_AA)
    for y in (Y_2M, Y_4M, Y_6M, Y_8M):
        cv2.line(img, (PITCH_L, y), (PITCH_R, y), (255, 255, 255), 1, cv2.LINE_AA)


def _draw_label_pill(img, text, cx, cy, bg_bgr, font_scale=0.52, thickness=2):
    """High-contrast label pill — readable on light zone colours."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    x1, y1 = cx - tw // 2 - 6, cy - th // 2 - 5
    x2, y2 = cx + tw // 2 + 6, cy + th // 2 + 5
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_bgr, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.putText(img, text, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_DUPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_zone_names_on_pitch(img, light=True):
    """Length names centred in each zone — dark pill for contrast."""
    for y1, y2, col, light_col, label in PITCH_ZONES:
        mid_y = (y1 + y2) // 2
        short = label.replace('BACK OF A ', 'B.O.A ')
        _draw_label_pill(img, short, CENTER_X, mid_y, col if not light else (60, 60, 120),
                         font_scale=0.48 if len(short) > 10 else 0.55)


def _draw_zone_names_right_column(img, bounces=None):
    """Right column coloured tags — like broadcast pitch map."""
    stats = zone_stats(bounces) if bounces else {z: 0 for z in ZONE_NAMES}
    cv2.rectangle(img, (LABEL_X - 4, PITCH_TOP), (MAP_W - 8, PITCH_TOP + 24), (80, 50, 30), -1)
    cv2.putText(img, 'BALLS (%)', (LABEL_X + 2, PITCH_TOP + 17), cv2.FONT_HERSHEY_DUPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    for y1, y2, col, _, label in PITCH_ZONES:
        mid_y = (y1 + y2) // 2
        pct = stats.get(label, 0)
        short = label.replace('BACK OF A ', 'B.O.A ')
        tag = f'{short}  {pct}%'
        cv2.rectangle(img, (LABEL_X, mid_y - 12), (MAP_W - 10, mid_y + 12), col, -1)
        cv2.putText(img, tag, (LABEL_X + 4, mid_y + 5), cv2.FONT_HERSHEY_DUPLEX,
                    0.38, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_zone_pct_on_pitch(img, bounces):
    """Show % count on right edge inside pitch."""
    stats = zone_stats(bounces)
    for y1, y2, col, _, label in PITCH_ZONES:
        mid_y = (y1 + y2) // 2
        pct = stats.get(label, 0)
        if pct <= 0:
            continue
        tag = f'{pct}%'
        cv2.rectangle(img, (PITCH_R - 42, mid_y - 9), (PITCH_R - 4, mid_y + 9), col, -1)
        cv2.putText(img, tag, (PITCH_R - 38, mid_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_distance_markers_on_pitch(img):
    """Distance labels inside left edge of pitch."""
    markers = [
        (PITCH_TOP + 12, 'STUMPS'),
        (Y_2M, '2M'),
        (Y_4M, '4M'),
        (Y_6M, '6M'),
        (Y_8M, '8M'),
        (Y_HALFWAY, 'HALFWAY'),
    ]
    for my, label in markers:
        cv2.line(img, (PITCH_L + 4, my), (PITCH_L + 30, my), (255, 255, 255), 2)
        cv2.putText(img, label, (PITCH_L + 34, my + 5), cv2.FONT_HERSHEY_DUPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, label, (PITCH_L + 34, my + 5), cv2.FONT_HERSHEY_DUPLEX,
                    0.45, (30, 30, 30), 2, cv2.LINE_AA)


def _draw_distance_markers(img):
    markers = [
        (PITCH_TOP + 12, 'STUMPS'),
        (Y_2M, '2M'),
        (Y_4M, '4M'),
        (Y_6M, '6M'),
        (Y_8M, '8M'),
        (Y_HALFWAY, 'HALFWAY'),
    ]
    for my, label in markers:
        cv2.line(img, (PITCH_L - 28, my), (PITCH_L - 6, my), (255, 255, 255), 1)
        cv2.putText(img, label, (PITCH_L - 88, my + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_zone_labels(img, bounces):
    _draw_zone_names_right_column(img, bounces)


def _draw_runs_legend(img):
    cv2.rectangle(img, (PITCH_L, MAP_H - 38), (PITCH_R, MAP_H - 10), (255, 255, 255), -1)
    lx = PITCH_L + 6
    for label, col in RUNS_LEGEND:
        cv2.circle(img, (lx + 5, MAP_H - 22), 5, col, -1, cv2.LINE_AA)
        cv2.putText(img, label, (lx + 12, MAP_H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (20, 20, 20), 1, cv2.LINE_AA)
        lx += 58


def map_point_to_video(px, py, H_inv):
    pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
    t = cv2.perspectiveTransform(pt, H_inv)
    return int(t[0, 0, 0]), int(t[0, 0, 1])


_TEMPLATE_CACHE = {}


def clear_template_cache():
    _TEMPLATE_CACHE.clear()


def _font_scale_for_zone(H_inv, y1, y2, frame_h):
    _, yt = map_point_to_video(CENTER_X, y1, H_inv)
    _, yb = map_point_to_video(CENTER_X, y2, H_inv)
    zone_h = max(12, abs(yb - yt))
    return max(0.55, min(1.4, zone_h / 22.0))


def _put_label_on_frame(frame, text, x, y, bg_bgr, font_scale=0.7):
    """Bold label with dark outline — readable on any background."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2)
    x1, y1 = x - 4, y - th - 6
    x2, y2 = x + tw + 4, y + 4
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_bgr, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)


def draw_distance_markers_on_video(frame, H_inv):
    """Distance markers — left edge only, single draw."""
    markers = [
        (PITCH_TOP + 12, 'STUMPS'),
        (Y_2M, '2M'),
        (Y_4M, '4M'),
        (Y_6M, '6M'),
        (Y_8M, '8M'),
        (Y_HALFWAY, 'HALF'),
    ]
    h, w = frame.shape[:2]
    for my, label in markers:
        lx, ly = map_point_to_video(PITCH_L + 20, my, H_inv)
        if 0 <= lx < w - 30 and 0 <= ly < h - 5:
            cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def draw_zone_labels_on_video(frame, H_inv, bounces=None):
    """Zone names — right edge only, one label per zone (no duplicates)."""
    h, w = frame.shape[:2]
    for y1, y2, col, _, label in PITCH_ZONES:
        mid_y = (y1 + y2) // 2
        short = label.replace('BACK OF A ', 'B.O.A ')
        rx, ry = map_point_to_video(PITCH_R - 5, mid_y, H_inv)
        if 0 <= rx < w - 10 and 0 <= ry < h - 5:
            cv2.putText(frame, short, (rx + 6, ry + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, short, (rx + 6, ry + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)


def create_full_pitch_overlay(bounces=None):
    """Colour zones only — all text drawn once on video after warp."""
    img = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)
    _draw_pitch_zones(img, light=True)
    mask = np.zeros((MAP_H, MAP_W), dtype=np.uint8)
    cv2.rectangle(mask, (PITCH_L, PITCH_TOP), (PITCH_R, PITCH_BOT), 255, -1)
    return img, mask


def create_light_pitch_overlay():
    return create_full_pitch_overlay(bounces=None)


def warp_pitch_overlay(frame, overlay, mask, H_inv, alpha=0.32):
    h, w = frame.shape[:2]
    warped = cv2.warpPerspective(overlay, H_inv, (w, h))
    warped_mask = cv2.warpPerspective(mask, H_inv, (w, h)).astype(np.float32) / 255.0
    warped_mask = warped_mask[:, :, np.newaxis]
    out = frame.astype(np.float32) * (1 - warped_mask * alpha) + warped.astype(np.float32) * (warped_mask * alpha)
    return out.astype(np.uint8)


def draw_light_bounce_dots(frame, bounces, use_video_coords=True):
    """Bounce markers on pitch — green=DOT, red=RUN, text beside dot."""
    for bounce in bounces:
        coords = bounce['coords'] if use_video_coords else bounce.get('map_coords', bounce['coords'])
        label = bounce.get('label') or bounce.get('type', 'DOTS')
        color = LIGHT_OUTCOME_BGR.get(label, LIGHT_OUTCOME_BGR['DOTS'])
        cx, cy = int(coords[0]), int(coords[1])
        cv2.circle(frame, (cx, cy), 14, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 14, (255, 255, 255), 2, cv2.LINE_AA)
        run_txt = {'DOTS': 'DOT', 'RUNS': 'RUN', 'BOUNDARIES': '4', 'WICKETS': 'OUT'}.get(label, 'DOT')
        tx, ty = cx + 18, cy + 5
        cv2.putText(frame, run_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, run_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)


def create_hawkeye_template(bowler_name='PITCH MAP'):
    img = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)
    img[:] = (40, 100, 40)
    cv2.rectangle(img, (100, 40), (MAP_W - 60, MAP_H - 50), (30, 85, 30), -1)

    _draw_pitch_zones(img, light=False)
    _draw_zone_names_on_pitch(img, light=False)
    _draw_distance_markers_on_pitch(img)
    _draw_zone_names_right_column(img, [])
    for sx in (CENTER_X - 16, CENTER_X, CENTER_X + 16):
        cv2.line(img, (sx, PITCH_TOP - 6), (sx, PITCH_TOP + 10), (0, 0, 200), 2)
    cv2.line(img, (PITCH_L, PITCH_TOP + 16), (PITCH_R, PITCH_TOP + 16), (240, 240, 240), 2)

    _draw_distance_markers(img)
    _draw_zone_labels(img, [])
    _draw_runs_legend(img)

    title = f'{bowler_name}'
    cv2.rectangle(img, (PITCH_L, MAP_H - 72), (PITCH_R, MAP_H - 48), (0, 200, 255), -1)
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.putText(img, title, ((PITCH_L + PITCH_R - tw) // 2, MAP_H - 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2, cv2.LINE_AA)
    return img


def render_pitch_map(bounces, bowler_name='PITCH MAP', base_img=None):
    img = base_img.copy() if base_img is not None else create_hawkeye_template(bowler_name)
    _draw_zone_labels(img, bounces)

    for ball in bounces:
        bx, by = ball['coords']
        b_type = ball.get('type', 'DOTS')
        length = ball.get('length') or classify_length_zone(by)
        color = OUTCOME_COLORS_BGR.get(b_type, OUTCOME_COLORS_BGR['DOTS'])
        bx = int(max(PITCH_L + 8, min(PITCH_R - 8, bx)))
        by = int(max(PITCH_TOP + 8, min(PITCH_BOT - 8, by)))
        cv2.circle(img, (bx, by), 9, color, -1, cv2.LINE_AA)
        cv2.circle(img, (bx, by), 9, (30, 30, 30), 1, cv2.LINE_AA)
        # tiny length tag near dot
        short_len = length.split()[0][:4]
        cv2.putText(img, short_len, (bx + 10, by + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def auto_pitch_quad(width, height):
    cx = width * 0.5
    return np.array([
        [cx - width * 0.10, height * 0.38],
        [cx + width * 0.10, height * 0.38],
        [cx - width * 0.26, height * 0.90],
        [cx + width * 0.26, height * 0.90],
    ], dtype=np.float32)


_TEMPLATE_CACHE = {}


def get_cached_template(title='PITCH MAP'):
    key = f'lengths-v3:{title}'
    if key not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[key] = create_hawkeye_template(title)
    return _TEMPLATE_CACHE[key]


def build_panel_image(bounces, title='PITCH MAP', panel_w=300):
    panel_w = max(280, panel_w)
    panel_h = int(panel_w * (MAP_H / MAP_W))
    # Always fresh render — never use stale cached template
    base = create_hawkeye_template(title)
    _draw_zone_labels(base, bounces)
    panel = render_pitch_map(bounces, bowler_name=title, base_img=base)
    return cv2.resize(panel, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)


def blit_panel(frame, panel, margin=12):
    h, w = frame.shape[:2]
    ph, pw = panel.shape[:2]
    x1 = max(margin, w - pw - margin)
    y1 = max(margin, h - ph - margin)
    x2, y2 = x1 + pw, y1 + ph
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (255, 255, 255), 2)
    frame[y1:y2, x1:x2] = panel
    return frame


def paste_hawkeye_panel(frame, bounces, title='PITCH MAP', panel_w=300):
    h, w = frame.shape[:2]
    panel_w = min(panel_w, w - 20)
    panel_h = int(panel_w * (MAP_H / MAP_W))
    if panel_h > h - 20:
        panel_h = h - 20
        panel_w = int(panel_h * (MAP_W / MAP_H))
    panel = build_panel_image(bounces, title, panel_w)
    return blit_panel(frame, panel)


def paste_hawkeye_panel_centered(frame, bounces, title='PITCH MAP', panel_img=None):
    h, w = frame.shape[:2]
    if panel_img is None:
        panel_w = int(min(w * 0.62, 480))
        panel_img = build_panel_image(bounces, title, panel_w)
    ph, pw = panel_img.shape[:2]
    x1 = (w - pw) // 2
    y1 = (h - ph) // 2
    x2, y2 = x1 + pw, y1 + ph
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)
    cv2.rectangle(frame, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), (255, 255, 255), 3)
    frame[y1:y2, x1:x2] = panel_img
    return frame


def draw_summary_banner(frame, text='PITCH MAP - ALL DELIVERIES'):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 40), (20, 50, 40), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 180), 2, cv2.LINE_AA)
    return frame
