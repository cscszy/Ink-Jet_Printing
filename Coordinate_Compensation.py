import sys
import re
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QLineEdit, QLabel, QMessageBox,
    QTextEdit, QHBoxLayout
)
from PyQt6.QtGui import QTextCharFormat, QColor, QFont


class GCodeModifier(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("G-code 原点偏移修正工具")
        self.setGeometry(200, 200, 800, 600)
        self.setFont(QFont("Simsun", 10))

        # 文件路径
        self.file_path = None
        self.original_lines = []
        self.modified_lines = []

        # 主布局
        main_layout = QVBoxLayout()

        # 文件读取按钮
        self.load_btn = QPushButton("读取 G-code 文件")
        self.load_btn.clicked.connect(self.load_file)
        main_layout.addWidget(self.load_btn)

        # 偏移量输入
        input_layout = QHBoxLayout()
        self.x_label = QLabel("X/x 修正量：")
        input_layout.addWidget(self.x_label)
        self.x_offset = QLineEdit()
        self.x_offset.setPlaceholderText("输入浮点数，例如 -1.23")
        input_layout.addWidget(self.x_offset)

        self.y_label = QLabel("Y/y 修正量：")
        input_layout.addWidget(self.y_label)
        self.y_offset = QLineEdit()
        self.y_offset.setPlaceholderText("输入浮点数，例如 2.5")
        input_layout.addWidget(self.y_offset)

        main_layout.addLayout(input_layout)

        # 生成预览按钮
        self.preview_btn = QPushButton("生成预览")
        self.preview_btn.clicked.connect(self.generate_preview)
        main_layout.addWidget(self.preview_btn)

        # 文本对比区
        text_layout = QHBoxLayout()
        self.original_text = QTextEdit()
        self.original_text.setReadOnly(True)
        self.original_text.setPlaceholderText("原始 G-code 内容")
        text_layout.addWidget(self.original_text)

        self.modified_text = QTextEdit()
        self.modified_text.setReadOnly(True)
        self.modified_text.setPlaceholderText("修改后的 G-code 内容")
        text_layout.addWidget(self.modified_text)

        main_layout.addLayout(text_layout)

        # 保存按钮
        self.save_btn = QPushButton("保存修改后的文件")
        self.save_btn.clicked.connect(self.save_file)
        main_layout.addWidget(self.save_btn)

        # 设置窗口
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # 让左右文本框的滚动条保持同步
        self.original_text.verticalScrollBar().valueChanged.connect(
            self.modified_text.verticalScrollBar().setValue
        )
        self.modified_text.verticalScrollBar().valueChanged.connect(
            self.original_text.verticalScrollBar().setValue
        )

    def load_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "选择 G-code 文件", "", "Text Files (*.txt)")
        if file_path:
            self.file_path = file_path
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.original_lines = f.readlines()
            self.original_text.setPlainText("".join(self.original_lines))
            self.modified_text.clear()
            QMessageBox.information(self, "成功", f"已选择文件：\n{file_path}")

    def generate_preview(self):
        if not self.original_lines:
            QMessageBox.warning(self, "错误", "请先选择一个 G-code 文件")
            return

        try:
            dx = float(self.x_offset.text())
            dy = float(self.y_offset.text())
        except ValueError:
            QMessageBox.warning(self, "错误", "请输入有效的浮点数偏移量")
            return

        self.modified_lines = []
        self.modified_text.clear()
        cursor = self.modified_text.textCursor()

        # 定义普通和高亮格式
        normal_format = QTextCharFormat()
        normal_format.setForeground(QColor("black"))
        normal_format.setFont(QFont("Simsun", 10))

        highlight_format = QTextCharFormat()
        highlight_format.setForeground(QColor("red"))

        # 正则匹配 X/Y/x/y 后的数值
        pattern = re.compile(r"([XYxy])(-?\d+\.\d+)")

        for orig_line in self.original_lines:
            pos = 0
            new_line = ""

            for match in pattern.finditer(orig_line):
                axis, value_str = match.groups()
                value = float(value_str)
                new_value = value + (dx if axis.lower() == "x" else dy)

                # 1. 插入前面没改的部分（普通格式）
                unchanged_text = orig_line[pos:match.start()]
                cursor.insertText(unchanged_text, normal_format)
                new_line += unchanged_text

                # 2. 插入 axis（普通格式）
                cursor.insertText(axis, normal_format)
                new_line += axis

                # 3. 插入修改后的数值（高亮格式）
                cursor.insertText(f"{new_value:.6f}", highlight_format)
                new_line += f"{new_value:.6f}"

                pos = match.end()

            # 插入最后剩下的部分（普通格式）
            rest_text = orig_line[pos:]
            cursor.insertText(rest_text, normal_format)
            new_line += rest_text

            self.modified_lines.append(new_line)

    def save_file(self):
        if not self.modified_lines:
            QMessageBox.warning(self, "错误", "请先生成预览再保存")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "保存修改后的文件", "", "Text Files (*.txt)")
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.writelines(self.modified_lines)
            QMessageBox.information(self, "成功", f"修改后的文件已保存至：\n{save_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GCodeModifier()
    window.show()
    sys.exit(app.exec())
