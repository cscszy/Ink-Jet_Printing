import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QLineEdit,
    QSpacerItem, QSizePolicy,
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams
from PyQt6.QtGui import QIntValidator
from PyQt6.QtGui import QFont

# ============================================================
# 工具函数：几何计算 & G-code 解析
# ============================================================

def arc_points(x0, y0, x1, y1, cx, cy, cw=True, step=0.05):
    """
    生成圆弧上的插值点。
    参数:
        x0, y0   起点坐标
        x1, y1   终点坐标
        cx, cy   圆心坐标
        cw       是否顺时针 (G02 为顺时针，G03 为逆时针)
        step     插值点间隔
    返回:
        [(x,y), (x,y), ...] 插值点列表
    """
    R = np.hypot(x0 - cx, y0 - cy)   # 半径
    theta0 = np.arctan2(y0 - cy, x0 - cx)
    theta1 = np.arctan2(y1 - cy, x1 - cx)

    # 起点与终点相同 → 画整圆
    if np.hypot(x1 - x0, y1 - y0) < 1e-9:
        arc_len = 2 * np.pi * R
        num = max(int(arc_len / step), 180)
        if cw:
            thetas = np.linspace(theta0, theta0 - 2*np.pi, num, endpoint=True)
        else:
            thetas = np.linspace(theta0, theta0 + 2*np.pi, num, endpoint=True)
        return [(cx + R*np.cos(t), cy + R*np.sin(t)) for t in thetas]

    # 起点和终点不同 → 画部分圆弧
    dtheta = theta1 - theta0
    if cw and dtheta > 0: dtheta -= 2*np.pi
    elif not cw and dtheta < 0: dtheta += 2*np.pi

    arc_len = abs(dtheta) * R
    num = max(int(arc_len / step), 32)
    ts = np.linspace(0.0, 1.0, num, endpoint=True)
    return [(cx + R*np.cos(theta0 + dtheta*t),
             cy + R*np.sin(theta0 + dtheta*t)) for t in ts]


def parse_gcode(lines, geom_step=0.05):
    """
    解析 G-code 文本，生成绘制片段 (segments)。
    每个片段包含点列、颜色、线型。
    """
    x, y = 0.0, 0.0
    segments = []
    for line in lines:
        parts = line.strip().split()
        if not parts: 
            continue
        cmd = parts[0].upper()   # 指令码，如 G01 / G28
        new_x, new_y = None, None
        cx, cy = None, None
        # 解析参数
        for p in parts[1:]:
            if p.startswith("X"): new_x = float(p[1:])
            elif p.startswith("Y"): new_y = float(p[1:])
            elif p.startswith("I"): cx = x + float(p[1:])   # 圆心相对坐标
            elif p.startswith("J"): cy = y + float(p[1:])

        if new_x is None: new_x = x
        if new_y is None: new_y = y

        points = []
        color = 'k'
        linestyle = '-'

        # 直线快速移动 (虚线)
        if cmd == "G00":
            points = [(x,y),(new_x,new_y)]
            color='0.7'; linestyle='--'

        # 直线插补 (实线)
        elif cmd=="G01":
            points=[(x,y),(new_x,new_y)]

        # 圆弧插补
        elif cmd in ("G02","G03"):
            cw=(cmd=="G02")
            if cx is not None and cy is not None:
                points = arc_points(x,y,new_x,new_y,cx,cy,cw=cw,step=geom_step)
            else:
                points = [(x,y),(new_x,new_y)]

        # 回到原点
        elif cmd=="G28":
            points = [(x,y),(0,0)]
            new_x, new_y = 0.0, 0.0

        # 程序结束
        elif cmd=="M02":
            break

        segments.append({'points': points,'color':color,'linestyle':linestyle})
        x,y=new_x,new_y
    return segments


def cumulative_lengths(points):
    """
    计算点列的累积长度数组。
    [0, d1, d1+d2, ...]
    """
    if not points: return np.array([0.0])
    L=[0.0]
    for i in range(1,len(points)):
        x0,y0=points[i-1]; x1,y1=points[i]
        L.append(L[-1]+np.hypot(x1-x0,y1-y0))
    return np.asarray(L)


def clip_segment_by_length(points,cumlen,s):
    """
    给定累计长度数组，截取到指定路径长度 s。
    返回: 已经绘制的点列、当前位置坐标
    """
    if s<=0: return [], points[0]
    if s>=cumlen[-1]: return points, points[-1]
    idx=int(np.searchsorted(cumlen,s,side='right'))
    idx=max(1,min(idx,len(points)-1))
    s0,s1=cumlen[idx-1],cumlen[idx]
    t=0.0 if s1==s0 else (s-s0)/(s1-s0)
    x0,y0=points[idx-1]; x1,y1=points[idx]
    xc,yc=(x0+t*(x1-x0),y0+t*(y1-y0))
    drawn=points[:idx]+[(xc,yc)]
    return drawn,(xc,yc)


# ============================================================
# PyQt6 GUI 动画类
# ============================================================

class GCodeAnimator(QWidget):
    """
    主窗口控件：
    - 显示 matplotlib 动画
    - 控制按钮（导入 / 播放 / 暂停 / 重置 / 退出）
    - 速度输入框
    """
    def __init__(self):
        super().__init__()
        # 路径相关
        self.segments=[]
        self.seg_cumlens=[]
        self.seg_lengths=[]
        self.seg_prefix=[]
        self.total_length=0.0

        # 动画参数
        self.speed=10.0   # mm/s
        self.dt=0.02      # 时间步长 s
        self.n_frames=1
        self.frame_idx_global=0
        self.ani=None

        # 绘图元素
        self.lines_objs=[]
        self.dot=None
        self.text_disp=None

        self.init_ui()

    def init_ui(self):
        """
        初始化 UI: matplotlib 画布 + 按钮 + 输入框
        """
        # 设置全局字体
        rcParams['font.family'] = 'Arial'
        rcParams['axes.unicode_minus'] = False
        self.setFont(QFont("Simsun", 10))

        layout = QVBoxLayout()

        # matplotlib 图
        self.fig, self.ax = plt.subplots(figsize=(6, 8))
        self.ax.set_title("G-code Printing Path", fontsize=20, fontweight="bold", pad=10)
        self.ax.set_xlabel("X (mm)", fontsize=20, fontweight="bold")
        self.ax.set_ylabel("Y (mm)", fontsize=20, fontweight="bold")
        for label in (self.ax.get_xticklabels() + self.ax.get_yticklabels()):
            label.set_fontsize(12)
            label.set_fontweight("bold")
        for spine in self.ax.spines.values():
            spine.set_linewidth(2)
        self.ax.tick_params(axis='both', width=2)

        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, stretch=1)

        # ---------------- 底部按钮行 ----------------
        hbox = QHBoxLayout()

        self.load_btn = QPushButton("导入txt文件")
        self.load_btn.clicked.connect(self.load_file)
        hbox.addWidget(self.load_btn)

        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.play_animation)
        hbox.addWidget(self.play_btn)

        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.pause_animation)
        hbox.addWidget(self.pause_btn)
        self.is_paused = False

        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset_animation)
        hbox.addWidget(self.reset_btn)

        self.exit_btn = QPushButton("退出")
        self.exit_btn.clicked.connect(QApplication.quit)
        hbox.addWidget(self.exit_btn)

        # 右侧留空，用来把速度输入框推到最右
        spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        hbox.addItem(spacer)

        # 速度输入框
        hbox.addWidget(QLabel("速度"))
        self.speed_input = QLineEdit(str(int(self.speed)))
        self.speed_input.setFixedWidth(40)
        self.speed_input.setValidator(QIntValidator(0, 50))  # 限制 [0, 50]
        self.speed_input.returnPressed.connect(self.change_speed_from_input)
        hbox.addWidget(self.speed_input)
        hbox.addWidget(QLabel("mm/s"))

        layout.addLayout(hbox, stretch=0)
        self.setLayout(layout)

    # ================== G-code 文件读取 ==================
    def load_file(self):
        """选择并加载 G-code 文件"""
        path,_=QFileDialog.getOpenFileName(self,"选择G-code文件","","Text Files (*.txt *.gcode);;All Files (*)")
        if not path: return
        with open(path,"r",encoding="utf-8") as f:
            lines=f.readlines()
        self.segments=parse_gcode(lines)
        self.compute_lengths()
        self.reset_animation()

        # 动态修改窗口标题
        window = self.window()
        if window is not None:
            import os
            filename = os.path.basename(path)
            window.setWindowTitle(f"G-code Printing Path - {filename}")

    # ================== 路径长度计算 ==================
    def compute_lengths(self):
        """预计算每条线段的累计长度，方便动画播放"""
        self.seg_cumlens=[]; self.seg_lengths=[]
        for seg in self.segments:
            pts=seg['points']
            if pts:
                L=cumulative_lengths(pts)
                self.seg_cumlens.append(L)
                self.seg_lengths.append(L[-1])
            else:
                self.seg_cumlens.append(np.array([0.0]))
                self.seg_lengths.append(0.0)
        self.seg_lengths=np.array(self.seg_lengths)
        self.seg_prefix=np.concatenate([[0.0],np.cumsum(self.seg_lengths)])
        self.total_length=self.seg_prefix[-1]
        self.n_frames=max(1,int(np.ceil(self.total_length/self.speed/self.dt)))

    # ================== 动画控制 ==================
    def reset_animation(self):
        """清空并重置绘图"""
        self.frame_idx_global = 0
        if self.ani is not None:
            if hasattr(self.ani, 'event_source') and self.ani.event_source is not None:
                self.ani.event_source.stop()
            self.ani = None

        # 清空画布
        self.ax.cla()
        self.ax.set_aspect('equal', adjustable='box')
        margin = 1
        all_pts = [p for seg in self.segments for p in seg['points'] if seg['points']]
        xs_all = [p[0] for p in all_pts] if all_pts else [0,1]
        ys_all = [p[1] for p in all_pts] if all_pts else [0,1]
        self.ax.set_xlim(min(xs_all)-margin, max(xs_all)+margin)
        self.ax.set_ylim(min(ys_all)-margin, max(ys_all)+2*margin)

        # 标题与标签
        self.ax.set_title("G-code Printing Path", fontsize=20, fontweight="bold", pad=10)
        self.ax.set_xlabel("X (mm)", fontsize=20, fontweight="bold")
        self.ax.set_ylabel("Y (mm)", fontsize=20, fontweight="bold")
        for label in (self.ax.get_xticklabels() + self.ax.get_yticklabels()):
            label.set_fontsize(12)
            label.set_fontweight("bold")

        # 初始化空线条对象
        self.lines_objs = []
        for seg in self.segments:
            if not seg['points']:
                self.lines_objs.append(None)
                continue
            line, = self.ax.plot([], [], linestyle=seg['linestyle'], color=seg['color'])
            self.lines_objs.append(line)

        # 红点 (当前打印位置)
        self.dot, = self.ax.plot([], [], 'ro', markersize=4)
        # 状态文字
        self.text_disp = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                    fontsize=10, va='top', ha='left')
        self.canvas.draw()

    def _apply_speed_from_ui(self):
        """从 QLineEdit 获取速度 (mm/s)，更新 self.speed"""
        txt = self.speed_input.text().strip()
        try:
            v = float(txt)
            if v <= 0:
                raise ValueError
        except ValueError:
            # 非法输入 -> 恢复为原速度
            self.speed_input.setText(str(int(self.speed)))
            v = self.speed
        self.speed = v

        # 速度变化 -> 更新帧数
        if self.total_length > 0:
            self.n_frames = max(1, int(np.ceil(self.total_length / self.speed / self.dt)))

    def change_speed_from_input(self):
        """当用户在文本框中按回车时触发"""
        self._apply_speed_from_ui()

    def play_animation(self):
        """播放动画"""
        if not self.segments:
            return
        self._apply_speed_from_ui()
        if self.ani is not None and getattr(self.ani, "event_source", None) is not None:
            self.ani.event_source.stop()
            self.ani = None
        self.frame_idx_global = 0
        self.ani = FuncAnimation(
            self.fig, self.update,
            frames=self.n_frames,
            interval=int(self.dt * 1000),
            blit=True,
            repeat=False
        )
        self.canvas.draw_idle()

    def pause_animation(self):
        """暂停 / 继续 动画"""
        if self.ani is None or getattr(self.ani, "event_source", None) is None:
            return
        if not self.is_paused:
            self.ani.event_source.stop()
            self.pause_btn.setText("继续")
            self.is_paused = True
        else:
            self.ani.event_source.start()
            self.pause_btn.setText("暂停")
            self.is_paused = False

    # ================== 动画更新 ==================
    def update(self,_):
        """动画每帧更新函数"""
        D=min(self.frame_idx_global*self.speed*self.dt,self.total_length)
        k=int(np.searchsorted(self.seg_prefix,D,side='right'))-1
        k=max(0,min(k,len(self.segments)-1))

        for i in range(len(self.lines_objs)):
            if self.lines_objs[i] is None: continue
            if i<k:   # 已完成
                pts=self.segments[i]['points']
                xs,ys=zip(*pts)
                self.lines_objs[i].set_data(xs,ys)
            elif i==k:  # 当前片段
                s_in_seg=D-self.seg_prefix[k]
                pts=self.segments[k]['points']
                cum=self.seg_cumlens[k]
                drawn_pts,head=clip_segment_by_length(pts,cum,s_in_seg)
                xs,ys=zip(*drawn_pts) if drawn_pts else ([],[])
                self.lines_objs[i].set_data(xs,ys)
            else:   # 未开始
                self.lines_objs[i].set_data([],[])

        if self.lines_objs[k] is not None:
            self.dot.set_data([head[0]],[head[1]])
        else:
            self.dot.set_data([],[])

        self.text_disp.set_text(f"Distance: {D:.1f} mm   Time: {D/self.speed:.1f} s")
        self.frame_idx_global+=1
        return [obj for obj in self.lines_objs if obj is not None]+[self.dot,self.text_disp]


# ============================================================
# 主程序入口
# ============================================================
if __name__=="__main__":
    app=QApplication(sys.argv)
    window=QMainWindow()
    animator=GCodeAnimator()
    window.setCentralWidget(animator)
    window.resize(800,1000)
    window.setWindowTitle("G-code Printing Path")
    window.show()
    sys.exit(app.exec())
