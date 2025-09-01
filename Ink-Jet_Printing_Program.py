import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QLineEdit,
    QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PyQt6.QtGui import QIntValidator

# -----------------------
# 复制原来的函数
# -----------------------
def arc_points(x0, y0, x1, y1, cx, cy, cw=True, step=0.05):
    R = np.hypot(x0 - cx, y0 - cy)
    theta0 = np.arctan2(y0 - cy, x0 - cx)
    theta1 = np.arctan2(y1 - cy, x1 - cx)
    if np.hypot(x1 - x0, y1 - y0) < 1e-9:
        arc_len = 2 * np.pi * R
        num = max(int(arc_len / step), 180)
        if cw:
            thetas = np.linspace(theta0, theta0 - 2*np.pi, num, endpoint=True)
        else:
            thetas = np.linspace(theta0, theta0 + 2*np.pi, num, endpoint=True)
        return [(cx + R*np.cos(t), cy + R*np.sin(t)) for t in thetas]
    dtheta = theta1 - theta0
    if cw and dtheta > 0: dtheta -= 2*np.pi
    elif not cw and dtheta < 0: dtheta += 2*np.pi
    arc_len = abs(dtheta) * R
    num = max(int(arc_len / step), 32)
    ts = np.linspace(0.0, 1.0, num, endpoint=True)
    return [(cx + R*np.cos(theta0 + dtheta*t),
             cy + R*np.sin(theta0 + dtheta*t)) for t in ts]

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
            if p.startswith("X") and len(p)>1: new_x = float(p[1:])
            elif p.startswith("Y") and len(p)>1: new_y = float(p[1:])
            elif p.startswith("x") and len(p)>1: cx = float(p[1:])
            elif p.startswith("y") and len(p)>1: cy = float(p[1:])
        if new_x is None: new_x = x
        if new_y is None: new_y = y
        points = []
        color = 'k'
        linestyle = '-'
        if cmd == "G00":
            points = [(x,y),(new_x,new_y)]
            color='0.7'; linestyle='--'
        elif cmd=="G01":
            points=[(x,y),(new_x,new_y)]
        elif cmd in ("G02","G03"):
            cw=(cmd=="G02")
            if cx is not None and cy is not None:
                points = arc_points(x,y,new_x,new_y,cx,cy,cw=cw,step=geom_step)
            else:
                points = [(x,y),(new_x,new_y)]
        elif cmd=="M02":
            break
        segments.append({'points': points,'color':color,'linestyle':linestyle})
        x,y=new_x,new_y
    return segments

def cumulative_lengths(points):
    if not points: return np.array([0.0])
    L=[0.0]
    for i in range(1,len(points)):
        x0,y0=points[i-1]; x1,y1=points[i]
        L.append(L[-1]+np.hypot(x1-x0,y1-y0))
    return np.asarray(L)

def clip_segment_by_length(points,cumlen,s):
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

# -----------------------
# PyQt6 GUI 动画类
# -----------------------
class GCodeAnimator(QWidget):
    def __init__(self):
        super().__init__()
        self.segments=[]
        self.seg_cumlens=[]
        self.seg_lengths=[]
        self.seg_prefix=[]
        self.total_length=0.0
        self.speed=10.0
        self.dt=0.02
        self.n_frames=1
        self.frame_idx_global=0
        self.ani=None
        self.lines_objs=[]
        self.dot=None
        self.text_disp=None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        # matplotlib canvas
        self.fig, self.ax = plt.subplots(figsize=(6, 8))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, stretch=1)   # 图像占满空间

        # 按钮与速度输入 (一行，速度在最右侧)
        hbox = QHBoxLayout()

        self.load_btn = QPushButton("导入G-code")
        self.load_btn.clicked.connect(self.load_file)
        hbox.addWidget(self.load_btn)

        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.play_animation)
        hbox.addWidget(self.play_btn)

        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset_animation)
        hbox.addWidget(self.reset_btn)

        spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        hbox.addItem(spacer)

        hbox.addWidget(QLabel("速度"))
        self.speed_input = QLineEdit(str(int(self.speed)))
        self.speed_input.setFixedWidth(40)
        self.speed_input.returnPressed.connect(self.change_speed_from_input)
        self.speed_input.setValidator(QIntValidator(0, 50))  # 限制输入范围在 [0, 50]
        self.speed_input.returnPressed.connect(self.change_speed_from_input)
        hbox.addWidget(self.speed_input)
        hbox.addWidget(QLabel("mm/s"))

        layout.addLayout(hbox, stretch=0)   # 底部按钮行不会被拉大
        self.setLayout(layout)



    def load_file(self):
        path,_=QFileDialog.getOpenFileName(self,"选择G-code文件","","Text Files (*.txt *.gcode);;All Files (*)")
        if not path: return
        with open(path,"r",encoding="utf-8") as f:
            lines=f.readlines()
        self.segments=parse_gcode(lines)
        self.compute_lengths()
        self.reset_animation()

    def compute_lengths(self):
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

    def reset_animation(self):
        self.frame_idx_global = 0
        if self.ani is not None:
            if hasattr(self.ani, 'event_source') and self.ani.event_source is not None:
                self.ani.event_source.stop()
            self.ani = None  # 清空旧动画
        # 后续清空 axes、绘制线条和红点
        self.ax.cla()
        self.ax.set_aspect('equal', adjustable='box')
        margin = 1
        all_pts = [p for seg in self.segments for p in seg['points'] if seg['points']]
        xs_all = [p[0] for p in all_pts] if all_pts else [0,1]
        ys_all = [p[1] for p in all_pts] if all_pts else [0,1]
        self.ax.set_xlim(min(xs_all)-margin, max(xs_all)+margin)
        self.ax.set_ylim(min(ys_all)-margin, max(ys_all)+2*margin)
        self.ax.set_title("G-code Printing Path")

        self.lines_objs = []
        for seg in self.segments:
            if not seg['points']:
                self.lines_objs.append(None)
                continue
            line, = self.ax.plot([], [], linestyle=seg['linestyle'], color=seg['color'])
            self.lines_objs.append(line)

        self.dot, = self.ax.plot([], [], 'ro', markersize=4)
        self.text_disp = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                    fontsize=10, va='top', ha='left')
        self.canvas.draw()

    def _apply_speed_from_ui(self):
        """从 QLineEdit 读取速度(mm/s)，更新 self.speed 与帧数"""
        txt = self.speed_input.text().strip()
        try:
            v = float(txt)
            if v <= 0:
                raise ValueError
        except ValueError:
            # 非法输入 -> 回退到当前有效值
            self.speed_input.setText(str(int(self.speed)))
            v = self.speed

        if v != self.speed:
            self.speed = v
        # 速度变化 -> 重新计算帧数（总时长 = total_length / speed）
        if self.total_length > 0:
            self.n_frames = max(1, int(np.ceil(self.total_length / self.speed / self.dt)))


    def change_speed_from_input(self):
        """当用户在文本框中按回车时调用"""
        self._apply_speed_from_ui()


    def play_animation(self):
        if not self.segments:
            return

        # 先应用文本框里的速度（即使用户没按回车）
        self._apply_speed_from_ui()

        # 停止旧动画
        if self.ani is not None and getattr(self.ani, "event_source", None) is not None:
            self.ani.event_source.stop()
            self.ani = None

        # 重置帧索引
        self.frame_idx_global = 0

        # 新动画
        self.ani = FuncAnimation(
            self.fig, self.update,
            frames=self.n_frames,
            interval=int(self.dt * 1000),  # 每帧间隔
            blit=True,
            repeat=False
        )
        self.canvas.draw_idle()



    def update(self,_):
        D=min(self.frame_idx_global*self.speed*self.dt,self.total_length)
        k=int(np.searchsorted(self.seg_prefix,D,side='right'))-1
        k=max(0,min(k,len(self.segments)-1))
        for i in range(len(self.lines_objs)):
            if self.lines_objs[i] is None: continue
            if i<k:
                pts=self.segments[i]['points']
                xs,ys=zip(*pts)
                self.lines_objs[i].set_data(xs,ys)
            elif i==k:
                s_in_seg=D-self.seg_prefix[k]
                pts=self.segments[k]['points']
                cum=self.seg_cumlens[k]
                drawn_pts,head=clip_segment_by_length(pts,cum,s_in_seg)
                xs,ys=zip(*drawn_pts) if drawn_pts else ([],[])
                self.lines_objs[i].set_data(xs,ys)
            else:
                self.lines_objs[i].set_data([],[])
        if self.lines_objs[k] is not None:
            self.dot.set_data([head[0]],[head[1]])
        else:
            self.dot.set_data([],[])
        self.text_disp.set_text(f"Distance: {D:.3f} mm   Time: {D/self.speed:.3f} s")
        self.frame_idx_global+=1
        return [obj for obj in self.lines_objs if obj is not None]+[self.dot,self.text_disp]

# -----------------------
# 主程序
# -----------------------
if __name__=="__main__":
    app=QApplication(sys.argv)
    window=QMainWindow()
    animator=GCodeAnimator()
    window.setCentralWidget(animator)
    window.resize(800,1000)
    window.show()
    sys.exit(app.exec())
