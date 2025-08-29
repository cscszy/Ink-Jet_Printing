import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================
# 高分辨率圆弧生成
# =========================
def arc_points(x0, y0, x1, y1, cx, cy, cw=True, step=0.05):
    R = np.hypot(x0 - cx, y0 - cy)
    theta0 = np.arctan2(y0 - cy, x0 - cx)
    theta1 = np.arctan2(y1 - cy, x1 - cx)

    if np.hypot(x1 - x0, y1 - y0) < 1e-9:  # 整圆
        arc_len = 2 * np.pi * R
        num = max(int(arc_len / step), 180)
        if cw:
            thetas = np.linspace(theta0, theta0 - 2*np.pi, num, endpoint=True)
        else:
            thetas = np.linspace(theta0, theta0 + 2*np.pi, num, endpoint=True)
        return [(cx + R*np.cos(t), cy + R*np.sin(t)) for t in thetas]

    dtheta = theta1 - theta0
    if cw and dtheta > 0:
        dtheta -= 2*np.pi
    elif not cw and dtheta < 0:
        dtheta += 2*np.pi

    arc_len = abs(dtheta) * R
    num = max(int(arc_len / step), 32)
    ts = np.linspace(0.0, 1.0, num, endpoint=True)
    return [(cx + R*np.cos(theta0 + dtheta*t),
             cy + R*np.sin(theta0 + dtheta*t)) for t in ts]

# =========================
# 解析 G-code
# =========================
def parse_gcode(lines, geom_step=0.05):
    x, y = 0.0, 0.0
    segments = []
    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        cmd = parts[0].upper()

        new_x, new_y = None, None
        cx, cy = None, None

        for p in parts[1:]:
            if p.startswith("X") and len(p) > 1:
                new_x = float(p[1:])
            elif p.startswith("Y") and len(p) > 1:
                new_y = float(p[1:])
            elif p.startswith("x") and len(p) > 1:
                cx = float(p[1:])
            elif p.startswith("y") and len(p) > 1:
                cy = float(p[1:])

        if new_x is None: new_x = x
        if new_y is None: new_y = y

        points = []
        color = 'k'
        linestyle = '-'

        if cmd == "G00":
            points = [(x, y), (new_x, new_y)]
            color = '0.7'
            linestyle = '--'
        elif cmd == "G01":
            points = [(x, y), (new_x, new_y)]
        elif cmd in ("G02", "G03"):
            cw = (cmd == "G02")
            if cx is not None and cy is not None:
                points = arc_points(x, y, new_x, new_y, cx, cy, cw=cw, step=geom_step)
            else:
                points = [(x, y), (new_x, new_y)]
        elif cmd == "M02":
            break

        segments.append({'points': points, 'color': color, 'linestyle': linestyle})
        x, y = new_x, new_y

    return segments

# =========================
# 累计长度 & 插值
# =========================
def cumulative_lengths(points):
    if not points: return np.array([0.0])
    L = [0.0]
    for i in range(1, len(points)):
        x0, y0 = points[i-1]
        x1, y1 = points[i]
        L.append(L[-1] + np.hypot(x1-x0, y1-y0))
    return np.asarray(L)

def clip_segment_by_length(points, cumlen, s):
    if s <= 0: return [], points[0]
    if s >= cumlen[-1]: return points, points[-1]
    idx = int(np.searchsorted(cumlen, s, side='right'))
    idx = max(1, min(idx, len(points)-1))
    s0, s1 = cumlen[idx-1], cumlen[idx]
    t = 0.0 if s1 == s0 else (s - s0)/(s1 - s0)
    x0, y0 = points[idx-1]
    x1, y1 = points[idx]
    xc, yc = (x0 + t*(x1-x0), y0 + t*(y1-y0))
    drawn = points[:idx] + [(xc, yc)]
    return drawn, (xc, yc)

# =========================
# 主流程
# =========================
file_path = r"C:/Users/cscszy/Desktop/Temp/蛇形电极1.txt"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

geom_step = 0.05
segments = parse_gcode(lines, geom_step=geom_step)

seg_cumlens, seg_lengths = [], []
for seg in segments:
    pts = seg['points']
    if pts:
        L = cumulative_lengths(pts)
        seg_cumlens.append(L)
        seg_lengths.append(L[-1])
    else:
        seg_cumlens.append(np.array([0.0]))
        seg_lengths.append(0.0)

seg_lengths = np.asarray(seg_lengths)
seg_prefix = np.concatenate([[0.0], np.cumsum(seg_lengths)])
total_length = seg_prefix[-1]

speed = 40.0  # mm/s
interval = 20  # ms
dt = interval / 1000.0
n_frames = max(1, int(np.ceil(total_length / speed / dt)))

# =========================
# 初始化 figure
# =========================
fig, ax = plt.subplots(figsize=(8,10))
ax.set_aspect('equal', adjustable='box')

lines_objs = []
dot = None
text_disp = None
frame_idx_global = 0
ani = None  # 动画对象

# =========================
# 刷新界面 & 重启动画
# =========================
def reset_figure():
    """刷新界面并重新启动动画"""
    global ax, lines_objs, dot, text_disp, frame_idx_global, ani

    # 停掉旧动画
    if ani is not None and ani.event_source is not None:
        ani.event_source.stop()

    ax.cla()
    ax.set_aspect('equal', adjustable='box')
    margin = 1
    all_pts = [p for seg in segments for p in seg['points'] if seg['points']]
    xs_all = [p[0] for p in all_pts] if all_pts else [0,1]
    ys_all = [p[1] for p in all_pts] if all_pts else [0,1]
    ax.set_xlim(min(xs_all)-margin, max(xs_all)+margin)
    ax.set_ylim(min(ys_all)-margin, max(ys_all)+2*margin)
    ax.set_title("G-code Printing Path")

    lines_objs.clear()
    for seg in segments:
        if not seg['points']:
            lines_objs.append(None)
            continue
        line, = ax.plot([], [], linestyle=seg['linestyle'], color=seg['color'])
        lines_objs.append(line)

    global dot, text_disp
    dot, = ax.plot([], [], 'ro', markersize=4)
    text_disp = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10, va='top', ha='left')

    frame_idx_global = 0

    # 重启动画
    ani = FuncAnimation(fig, update, frames=n_frames, interval=20, blit=True, repeat=False)
    fig.canvas.draw_idle()

# =========================
# 动画更新
# =========================
def update(_):
    global frame_idx_global

    D = min(frame_idx_global * speed * dt, total_length)
    k = int(np.searchsorted(seg_prefix, D, side='right')) - 1
    k = max(0, min(k, len(segments)-1))

    for i in range(len(lines_objs)):
        if lines_objs[i] is None:
            continue
        if i < k:
            pts = segments[i]['points']
            xs, ys = zip(*pts)
            lines_objs[i].set_data(xs, ys)
        elif i == k:
            s_in_seg = D - seg_prefix[k]
            pts = segments[k]['points']
            cum = seg_cumlens[k]
            drawn_pts, head = clip_segment_by_length(pts, cum, s_in_seg)
            xs, ys = zip(*drawn_pts) if drawn_pts else ([], [])
            lines_objs[i].set_data(xs, ys)
        else:
            lines_objs[i].set_data([], [])

    # 红点
    if lines_objs[k] is not None:
        dot.set_data([head[0]], [head[1]])
    else:
        dot.set_data([], [])

    text_disp.set_text(f"Distance: {D:.3f} mm   Time: {D/speed:.3f} s")
    frame_idx_global += 1
    return [obj for obj in lines_objs if obj is not None] + [dot, text_disp]

# =========================
# 首次初始化
# =========================
reset_figure()

# =========================
# 键盘事件
# =========================
def on_key(event):
    if event.key == 'r':
        reset_figure()  # 刷新并重绘动画
    elif event.key == 'q':
        plt.close(fig)

fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()
