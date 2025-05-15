import sys
import pandas as pd
import re
import os
import threading
import json
import logging
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox, QDialog, QDialogButtonBox, QFormLayout, QTextEdit, QComboBox
from PyQt6.QtCore import Qt, QEvent, QRegularExpression, pyqtSignal, QObject
from PyQt6.QtGui import QIcon, QRegularExpressionValidator
from openai import OpenAI
import jieba
from concurrent.futures import ThreadPoolExecutor

# 启动时检查并生成 stopwords.txt
STOPWORDS_FILE = os.path.join(os.path.dirname(sys.argv[0]), "stopwords.txt")
# 定义默认停用词列表（250 个）
DEFAULT_STOPWORDS = [
    '的', '是', '了', '在', '和',
    '有', '就', '你', '我', '他',
    '她', '它', '我们', '你们', '他们',
    '这', '那', '之', '也', '都',
    '不', '很', '会', '来', '去',
    '说', '想', '看', '能', '好',
    '一个', '这个', '那个', '这样', '那么',
    '所以', '因为', '但是', '如果', '虽然',
    '可是', '而且', '然后', '还是', '或者',
    '已经', '知道', '觉得', '现在', '时候',
    '可以', '没有', '怎么', '什么', '为什么',
    '怎么样', '一点', '一些', '很多', '多少',
    '太', '真', '还', '再', '就',
    '又', '才', '可', '与', '及',
    '跟', '被', '给', '让', '把',
    '从', '到', '向', '对于', '关于',
    '除了', '以外', '之外', '的话', '要是',
    '只要', '一边', '一面', '啊', '哦',
    '嗯', '呀', '吧', '呢', '啦',
    '嘛', '哈', '嘿', '哟', '哇',
    # 之前新增的 50 个
    '对', '比', '像', '似', '乎',
    '得', '着', '过', '完', '下',
    '上', '里', '外', '内', '间',
    '前', '后', '左', '右', '中',
    '嘛', '咯', '哒', '嘞', '呗',
    '哎呀', '哎哟', '嘿嘿', '哈哈哈', '嘻嘻',
    '嗯嗯', '哦哦', '啊啊', '哇塞', '天哪',
    '咋', '咋样', '啥', '咋办', '咋回事',
    '嘿哟', '哟呵', '嗷嗷', '嘤嘤', '哼哼',
    '啧', '呦', '嗳', '噜', '唷',
    # 新增 100 个停用词
    '嘛', '哩', '喏', '啵', '咧',
    '哪', '哪儿', '哪样', '咋啦', '咋了',
    '咋整', '咋地', '咋呼', '咋会', '咋呢',
    '嘿哈', '嘿呀', '哟喂', '嗷唠', '嘤咛',
    '唧唧', '咕咕', '嘀咕', '啪啪', '咔咔',
    '叮咚', '哒哒', '嗒嗒', '嗡嗡', '嘤嗡',
    '嗷呜', '嗷哟', '嗷嘿', '嗷吼', '嗷呀',
    '呸呸', '嘘嘘', '啧啧', '嘟嘟', '咕嘟',
    '咚咚', '叮当', '哐当', '咣当', '咔嚓',
    '咔嗒', '啪嗒', '啪啦', '哗啦', '哗哗',
    '滴滴', '答答', '嘀嗒', '嘀哒', '滴答',
    '哩哩', '啦啦', '唧唧', '喳喳', '叽叽',
    '哇哇', '啊啊', '哦呀', '哦呵', '哦哟',
    '嗯哼', '嗯哪', '嗯呐', '嗯哒', '嗯啦',
    '呀哈', '呀嘿', '呀呼', '呀哟', '呀呀',
    '嘿唷', '嘿呦', '嘿嗨', '嘿啦', '嘿哒',
    '哒啦', '哒嘛', '哒哟', '哒哈', '哒嘿',
    '啦哈', '啦嘿', '啦哒', '啦嘛', '啦哟',
    '嘛哈', '嘛嘿', '嘛哒', '嘛啦', '嘛哟'
]

if not os.path.exists(STOPWORDS_FILE):
    try:
        with open(STOPWORDS_FILE, 'w', encoding='utf-8') as f:
            f.write('\n'.join(DEFAULT_STOPWORDS))
        logging.info(f"已生成默认 stopwords.txt 文件：{STOPWORDS_FILE}")
    except Exception as e:
        logging.error(f"无法生成 stopwords.txt 文件：{str(e)}")

# 加载停用词
try:
    with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
        CHINESE_STOP_WORDS = set(f.read().splitlines())
except FileNotFoundError:
    CHINESE_STOP_WORDS = set(DEFAULT_STOPWORDS)
    logging.warning("未找到 stopwords.txt，使用默认中文停用词列表。")

# 初始化 jieba 词典
jieba.add_word("工业设计")
jieba.add_word("交互设计")
jieba.add_word("工程设计")
jieba.add_word("李小龙故居")
jieba.add_word("艺术街区")

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConnectionTestCompleteEvent(QEvent):
    def __init__(self):
        super().__init__(QEvent.Type(QEvent.Type.User + 5))

class ConnectionTestErrorEvent(QEvent):
    def __init__(self, error_msg):
        super().__init__(QEvent.Type(QEvent.Type.User + 6))
        self.error_msg = error_msg

class FilterSignals(QObject):
    error = pyqtSignal(str)
    finished = pyqtSignal(int)

class StopwordsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("编辑停用词")
        self.setGeometry(400, 400, 400, 300)
        self.layout = QVBoxLayout()

        self.label = QLabel("请输入停用词，每行一个：")
        self.layout.addWidget(self.label)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("每行一个停用词，例如：\n的\n是\n了")
        # 加载当前停用词
        self.text_edit.setText('\n'.join(CHINESE_STOP_WORDS))
        self.layout.addWidget(self.text_edit)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

        self.setLayout(self.layout)

    def get_stopwords(self):
        return [word.strip() for word in self.text_edit.toPlainText().splitlines() if word.strip()]

class SettingsDialog(QDialog):
    def __init__(self, parent=None, available_columns=None):
        super().__init__(parent)
        self.available_columns = available_columns or []
        self.setWindowTitle("设置")
        self.setGeometry(400, 400, 400, 550)
        self.layout = QFormLayout()

        self.api_key_label = QLabel("DeepSeek API Key:")
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("请输入 DeepSeek API Key")
        self.layout.addRow(self.api_key_label, self.api_key_input)

        self.test_button = QPushButton("测试连接")
        self.test_button.clicked.connect(self.test_connection)
        self.layout.addRow(self.test_button)

        self.prompt_label = QLabel("DeepSeek 提问方向:")
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("请输入 DeepSeek 提问方向（例如：请帮我提炼关于设计、艺术设计、交互设计、工程设计、工业设计相关的正向关键词，以及无关的生物、化学、人类学等反向关键词）")
        self.prompt_input.setFixedHeight(100)
        self.layout.addRow(self.prompt_label, self.prompt_input)

        self.positive_count_label = QLabel("正向关键词数量:")
        self.positive_count_input = QLineEdit()
        self.positive_count_input.setPlaceholderText("请输入正向关键词数量（数字）")
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d+$'))
        self.positive_count_input.setValidator(validator)
        self.layout.addRow(self.positive_count_label, self.positive_count_input)

        self.negative_count_label = QLabel("反向关键词数量:")
        self.negative_count_input = QLineEdit()
        self.negative_count_input.setPlaceholderText("请输入反向关键词数量（数字）")
        self.negative_count_input.setValidator(validator)
        self.layout.addRow(self.negative_count_label, self.negative_count_input)

        self.timeout_label = QLabel("API 超时时间（秒）:")
        self.timeout_input = QLineEdit()
        self.timeout_input.setPlaceholderText("请输入 API 超时时间（10-300 秒）")
        self.timeout_input.setValidator(validator)
        self.layout.addRow(self.timeout_label, self.timeout_input)

        self.language_label = QLabel("文本语言:")
        self.language_input = QComboBox()
        self.language_input.addItems(["英文", "中文"])
        self.layout.addRow(self.language_label, self.language_input)

        self.min_token_length_label = QLabel("分词最小长度:")
        self.min_token_length_input = QLineEdit()
        self.min_token_length_input.setPlaceholderText("请输入分词最小长度（1-5）")
        self.min_token_length_input.setValidator(QRegularExpressionValidator(QRegularExpression(r'^[1-5]$')))
        self.layout.addRow(self.min_token_length_label, self.min_token_length_input)

        self.max_token_length_label = QLabel("分词最大长度:")
        self.max_token_length_input = QLineEdit()
        self.max_token_length_input.setPlaceholderText("请输入分词最大长度（1-5）")
        self.max_token_length_input.setValidator(QRegularExpressionValidator(QRegularExpression(r'^[1-5]$')))
        self.layout.addRow(self.max_token_length_label, self.max_token_length_input)

        self.batch_size_label = QLabel("API 每次处理行数:")
        self.batch_size_input = QLineEdit()
        self.batch_size_input.setPlaceholderText("请输入行数（1-1000，建议 100）")
        validator = QRegularExpressionValidator(QRegularExpression(r'^[1-9]\d{0,2}$|^1000$'))
        self.batch_size_input.setValidator(validator)
        self.layout.addRow(self.batch_size_label, self.batch_size_input)

        self.concurrent_batches_label = QLabel("并行批次数:")
        self.concurrent_batches_input = QLineEdit()
        self.concurrent_batches_input.setPlaceholderText("请输入并行批次（1-10，建议 4）")
        validator = QRegularExpressionValidator(QRegularExpression(r'^[1-9]$|^10$'))
        self.concurrent_batches_input.setValidator(validator)
        self.layout.addRow(self.concurrent_batches_label, self.concurrent_batches_input)

        self.stopwords_button = QPushButton("编辑停用词")
        self.stopwords_button.clicked.connect(self.edit_stopwords)
        self.layout.addRow(self.stopwords_button)

        self.filter_columns_label = QLabel("筛选列（列名）：")
        self.filter_columns_input = QLineEdit()
        placeholder = "输入列名，英文逗号分隔，例如：Title,Abstract（留空使用所有列，无需引号）"
        if self.available_columns:
            placeholder += f"\n可用列名：{', '.join(self.available_columns)}"
        self.filter_columns_input.setPlaceholderText(placeholder)
        self.filter_columns_input.setToolTip("请输入 Excel 文件中的列名，多个列名用英文逗号分隔（,），无需引号，留空使用所有列")
        self.layout.addRow(self.filter_columns_label, self.filter_columns_input)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept_with_validation)
        self.buttons.rejected.connect(self.reject)
        self.layout.addRow(self.buttons)

        self.load_settings()
        self.setLayout(self.layout)
        self.msg_box = None

    def edit_stopwords(self):
        dialog = StopwordsDialog(self)
        if dialog.exec():
            new_stopwords = dialog.get_stopwords()
            global CHINESE_STOP_WORDS
            CHINESE_STOP_WORDS = set(new_stopwords)
            try:
                with open(STOPWORDS_FILE, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(new_stopwords))
                QMessageBox.information(self, "成功", "停用词保存成功！")
            except Exception as e:
                logger.error(f"保存停用词失败：{str(e)}")
                QMessageBox.critical(self, "错误", f"保存停用词失败：{str(e)}")

    def load_settings(self):
        settings_file = os.path.join(os.path.dirname(sys.argv[0]), "settings.json")
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                self.api_key_input.setText(settings.get("api_key", ""))
                self.prompt_input.setText(settings.get("prompt", ""))
                self.positive_count_input.setText(str(settings.get("positive_count", "")))
                self.negative_count_input.setText(str(settings.get("negative_count", "")))
                self.timeout_input.setText(str(settings.get("timeout", "")))
                self.language_input.setCurrentText(settings.get("language", "中文"))
                self.min_token_length_input.setText(str(settings.get("min_token_length", "")))
                self.max_token_length_input.setText(str(settings.get("max_token_length", "")))
                self.batch_size_input.setText(str(settings.get("batch_size", "")))
                self.concurrent_batches_input.setText(str(settings.get("concurrent_batches", "")))
                self.filter_columns_input.setText(settings.get("filter_columns", ""))
            except Exception as e:
                logger.error(f"加载设置失败：{str(e)}")
                QMessageBox.warning(self, "错误", f"加载设置失败：{str(e)}")

    def validate_filter_columns(self, text, columns=None):
        if not text.strip():
            return True, ""
        if columns is None:
            return True, ""
        text = text.replace('，', ',')
        column_names = [col.strip().strip("'").strip('"') for col in text.split(',')]
        if '，' in text and not ',' in text:
            return False, "请使用英文逗号（,）分隔列名，而不是中文逗号（，）"
        if any(col.isdigit() for col in column_names):
            return False, "请输入列名而非列号（例如：Title,Abstract，无需引号）"
        columns_lower = [col.lower() for col in columns]
        invalid_columns = [col for col in column_names if col.lower() not in columns_lower]
        if invalid_columns:
            return False, f"无效列名：{invalid_columns}（可用列名：{', '.join(columns)}）"
        return True, ""

    def accept_with_validation(self):
        filter_columns = self.filter_columns_input.text().strip()
        valid, error_msg = self.validate_filter_columns(filter_columns, self.available_columns)
        if not valid:
            QMessageBox.warning(self, "错误", error_msg)
            return
        self.accept()

    def save_settings(self):
        timeout = self.timeout_input.text().strip()
        timeout_value = int(timeout) if timeout else 60
        timeout_value = max(10, min(300, timeout_value))
        min_token_length = self.min_token_length_input.text().strip()
        min_token_length_value = int(min_token_length) if min_token_length else 1
        min_token_length_value = max(1, min(5, min_token_length_value))
        max_token_length = self.max_token_length_input.text().strip()
        max_token_length_value = int(max_token_length) if max_token_length else 5
        max_token_length_value = max(min_token_length_value, min(5, max_token_length_value))
        batch_size = self.batch_size_input.text().strip()
        batch_size_value = int(batch_size) if batch_size else 100
        batch_size_value = max(1, min(1000, batch_size_value))
        concurrent_batches = self.concurrent_batches_input.text().strip()
        concurrent_batches_value = int(concurrent_batches) if concurrent_batches else 4
        concurrent_batches_value = max(1, min(10, concurrent_batches_value))
        settings = {
            "api_key": self.api_key_input.text().strip(),
            "prompt": self.prompt_input.toPlainText().strip(),
            "positive_count": int(self.positive_count_input.text().strip()) if self.positive_count_input.text().strip() else 15,
            "negative_count": int(self.negative_count_input.text().strip()) if self.negative_count_input.text().strip() else 60,
            "timeout": timeout_value,
            "language": self.language_input.currentText(),
            "min_token_length": min_token_length_value,
            "max_token_length": max_token_length_value,
            "batch_size": batch_size_value,
            "concurrent_batches": concurrent_batches_value,
            "filter_columns": self.filter_columns_input.text().strip()
        }
        settings_file = os.path.join(os.path.dirname(sys.argv[0]), "settings.json")
        try:
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "成功", "设置保存成功！")
        except Exception as e:
            logger.error(f"保存设置失败：{str(e)}")
            QMessageBox.critical(self, "错误", f"保存设置失败：{str(e)}")

    def test_connection(self):
        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "提示", "请输入 API Key！")
            return
        timeout = self.timeout_input.text().strip()
        timeout_value = int(timeout) if timeout else 60
        timeout_value = max(10, min(300, timeout_value))
        self.msg_box = QMessageBox(self)
        self.msg_box.setWindowTitle("测试连接")
        self.msg_box.setText("正在连接...")
        self.msg_box.setStandardButtons(QMessageBox.StandardButton.NoButton)
        self.msg_box.show()

        def run_test():
            try:
                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": "Hello"},
                    ],
                    stream=False,
                    timeout=timeout_value
                )
                if response.choices and response.choices[0].message.content:
                    QApplication.instance().postEvent(self, ConnectionTestCompleteEvent())
                else:
                    QApplication.instance().postEvent(self, ConnectionTestErrorEvent("DeepSeek API 返回空响应"))
            except Exception as e:
                QApplication.instance().postEvent(self, ConnectionTestErrorEvent(f"连接失败：{str(e)}"))

        threading.Thread(target=run_test, daemon=True).start()

    def customEvent(self, event):
        if isinstance(event, ConnectionTestCompleteEvent):
            if self.msg_box:
                self.msg_box.setText("连接成功！")
                self.msg_box.setWindowTitle("成功")
                self.msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        elif isinstance(event, ConnectionTestErrorEvent):
            if self.msg_box:
                self.msg_box.setText(event.error_msg)
                self.msg_box.setWindowTitle("失败")
                self.msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)

    def get_settings(self):
        positive_count = self.positive_count_input.text().strip()
        negative_count = self.negative_count_input.text().strip()
        timeout = self.timeout_input.text().strip()
        min_token_length = self.min_token_length_input.text().strip()
        max_token_length = self.max_token_length_input.text().strip()
        batch_size = self.batch_size_input.text().strip()
        concurrent_batches = self.concurrent_batches_input.text().strip()
        settings = {
            "api_key": self.api_key_input.text().strip(),
            "prompt": self.prompt_input.toPlainText().strip() or (
                "请从以下文本中提取与设计相关的正向关键词（如‘工业设计’, ‘李小龙故居’, ‘艺术设计’, ‘软件工程’, ‘交互设计’, ‘艺术街区’等，优先保留完整词组，词长 2-5 个字符），"
                "以及无关的反向关键词（如‘生物技术’, ‘生物工程’, ‘生物化学’, ‘生物信息学’, ‘化学’等）。"
                "正向关键词需与设计、艺术、工程、机器人、软件工程相关，至少 {self.positive_count} 个。"
                "反向关键词需与生物、化学、医学、物理等无关领域相关，至少 {self.negative_count} 个。"
                "返回 JSON 格式，包含 'positive_keywords' 和 'negative_keywords' 两个字段，语言为 {self.language}。"
                "示例：{'positive_keywords': ['工业设计', '李小龙故居'], 'negative_keywords': ['生物技术', '化学']}"
            ),
            "positive_count": int(positive_count) if positive_count else 15,
            "negative_count": int(negative_count) if negative_count else 60,
            "timeout": int(timeout) if timeout else 60,
            "language": self.language_input.currentText(),
            "min_token_length": int(min_token_length) if min_token_length else 1,
            "max_token_length": int(max_token_length) if max_token_length else 5,
            "batch_size": int(batch_size) if batch_size else 100,
            "concurrent_batches": int(concurrent_batches) if concurrent_batches else 4,
            "filter_columns": self.filter_columns_input.text().strip()
        }
        self.save_settings()
        return settings

class DesignFilterTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("筛选器")
        self.setGeometry(300, 300, 600, 300)

        # 兼容 PyInstaller 打包后的路径
        def resource_path(relative_path):
            if hasattr(sys, '_MEIPASS'):
                return os.path.join(sys._MEIPASS, relative_path)
            return os.path.join(os.path.dirname(sys.argv[0]), relative_path)
        
        # 使用兼容路径加载图标
        self.setWindowIcon(QIcon(resource_path('filter.ico')))

        self.settings_action = QPushButton()
        self.settings_action.setIcon(QIcon(resource_path('setting.ico')))
        self.settings_action.setToolTip("设置 DeepSeek API")
        self.settings_action.setFixedSize(30, 30)
        self.settings_action.clicked.connect(self.show_settings_dialog)
        self.setMenuWidget(self.settings_action)

        self.api_key = ""
        self.prompt = ""
        self.positive_count = 15
        self.negative_count = 60
        self.timeout = 60
        self.language = "中文"
        self.min_token_length = 1
        self.max_token_length = 5
        self.batch_size = 100
        self.concurrent_batches = 4
        self.filter_columns = ""
        self.load_settings()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.include_keyword_layout = QHBoxLayout()
        self.include_keyword_label = QLabel("正向关键词：")
        self.include_keyword_input = QLineEdit()
        self.include_keyword_input.setPlaceholderText("输入或等待 AI 提取，例如 design,industrial（逗号分隔，无需引号）")
        self.include_keyword_layout.addWidget(self.include_keyword_label)
        self.include_keyword_layout.addWidget(self.include_keyword_input)
        self.layout.addLayout(self.include_keyword_layout)

        self.exclude_keyword_layout = QHBoxLayout()
        self.exclude_keyword_label = QLabel("反向关键词：")
        self.exclude_keyword_input = QLineEdit()
        self.exclude_keyword_input.setPlaceholderText("输入或等待 AI 提取，例如 biology,medicine（逗号分隔，无需引号）")
        self.exclude_keyword_layout.addWidget(self.exclude_keyword_label)
        self.exclude_keyword_layout.addWidget(self.exclude_keyword_input)
        self.layout.addLayout(self.exclude_keyword_layout)

        self.button_layout = QHBoxLayout()
        self.import_button = QPushButton("导入")
        self.import_button.clicked.connect(self.import_file)
        self.extract_local_button = QPushButton("本地关键词提取")
        self.extract_local_button.clicked.connect(self.extract_keywords_locally)
        self.ai_keyword_button = QPushButton("AI关键词")
        self.ai_keyword_button.clicked.connect(self.extract_keywords_with_deepseek)
        self.filter_button = QPushButton("筛选")
        self.filter_button.clicked.connect(self.start_filtering)
        self.save_keywords_button = QPushButton("保存关键词")
        self.save_keywords_button.clicked.connect(self.save_keywords)
        self.load_keywords_button = QPushButton("加载关键词")
        self.load_keywords_button.clicked.connect(self.load_keywords)
        self.export_button = QPushButton("导出")
        self.export_button.clicked.connect(self.export_file)
        self.button_layout.addWidget(self.import_button)
        self.button_layout.addWidget(self.extract_local_button)
        self.button_layout.addWidget(self.ai_keyword_button)
        self.button_layout.addWidget(self.filter_button)
        self.button_layout.addWidget(self.save_keywords_button)
        self.button_layout.addWidget(self.load_keywords_button)
        self.button_layout.addWidget(self.export_button)
        self.layout.addLayout(self.button_layout)

        self.df = None
        self.filtered_df = None
        self.import_file_path = None
        self.column_names = []
        self.processing = False
        self.local_keywords = []
        self.keyword_tfidf_scores = {}
        self.filter_signals = FilterSignals()
        self.filter_signals.error.connect(self.on_filter_error)
        self.filter_signals.finished.connect(self.on_filter_finished)

        self.extract_local_button.setEnabled(False)
        self.ai_keyword_button.setEnabled(False)
        self.filter_button.setEnabled(False)
        self.save_keywords_button.setEnabled(False)
        self.load_keywords_button.setEnabled(True)

    def load_settings(self):
        settings_file = os.path.join(os.path.dirname(sys.argv[0]), "settings.json")
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                self.api_key = settings.get("api_key", "")
                self.prompt = settings.get("prompt", "")
                self.positive_count = settings.get("positive_count", 15)
                self.negative_count = settings.get("negative_count", 60)
                self.timeout = settings.get("timeout", 60)
                self.language = settings.get("language", "中文")
                self.min_token_length = settings.get("min_token_length", 1)
                self.max_token_length = settings.get("max_token_length", 5)
                self.batch_size = settings.get("batch_size", 100)
                self.concurrent_batches = settings.get("concurrent_batches", 4)
                self.filter_columns = settings.get("filter_columns", "")
                self.timeout = max(10, min(300, self.timeout))
                self.batch_size = max(1, min(1000, self.batch_size))
                self.concurrent_batches = max(1, min(10, self.concurrent_batches))
            except Exception as e:
                logger.error(f"加载设置失败：{str(e)}")
                QMessageBox.warning(self, "错误", f"加载设置失败：{str(e)}")

    def show_settings_dialog(self):
        dialog = SettingsDialog(self, self.column_names)
        if dialog.exec():
            settings = dialog.get_settings()
            self.api_key = settings["api_key"]
            self.prompt = settings["prompt"]
            self.positive_count = settings["positive_count"]
            self.negative_count = settings["negative_count"]
            self.timeout = settings["timeout"]
            self.language = settings["language"]
            self.min_token_length = settings["min_token_length"]
            self.max_token_length = settings["max_token_length"]
            self.batch_size = settings["batch_size"]
            self.concurrent_batches = settings["concurrent_batches"]
            self.filter_columns = settings["filter_columns"]
            self.ai_keyword_button.setEnabled(bool(self.api_key))

    def import_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择Excel文件", "", "Excel Files (*.xlsx *.xls)")
        if not file_path:
            return
        try:
            self.df = pd.read_excel(file_path)
            self.df = self.df.fillna('').astype(str).apply(lambda x: x.str.lower())
            self.import_file_path = file_path
            self.column_names = self.df.columns.tolist()
            self.filtered_df = None
            self.local_keywords = []
            self.keyword_tfidf_scores = {}
            self.include_keyword_input.clear()
            self.exclude_keyword_input.clear()
            self.extract_local_button.setText("本地关键词提取")
            self.ai_keyword_button.setText("AI关键词")
            self.filter_button.setText("筛选")
            self.extract_local_button.setEnabled(True)
            self.ai_keyword_button.setEnabled(bool(self.api_key))
            self.filter_button.setEnabled(True)
            self.save_keywords_button.setEnabled(True)
            column_info = f"列名（共 {len(self.column_names)} 列）：\n" + ", ".join(f"{i+1}: {col}" for i, col in enumerate(self.column_names))
            QMessageBox.information(self, "提示", f"导入成功，共 {len(self.df)} 条数据！\n\n{column_info}")
        except Exception as e:
            logger.error(f"无法读取文件：{str(e)}")
            QMessageBox.critical(self, "错误", f"无法读取文件：{str(e)}")
            self.df = None
            self.import_file_path = None
            self.column_names = []
            self.extract_local_button.setEnabled(False)
            self.ai_keyword_button.setEnabled(False)
            self.filter_button.setEnabled(False)
            self.save_keywords_button.setEnabled(False)

    def get_filter_columns(self):
        if not self.filter_columns.strip():
            return None
        try:
            filter_columns = self.filter_columns.replace('，', ',')
            column_names = [col.strip().strip("'").strip('"') for col in filter_columns.split(',')]
            columns = []
            column_names_lower = [col.lower() for col in self.column_names]
            for col in column_names:
                if col.lower() not in column_names_lower:
                    raise ValueError(f"无效列名：{col}")
                index = column_names_lower.index(col.lower())
                columns.append(index)
            return columns
        except ValueError as e:
            logger.error(f"筛选列配置错误：{str(e)}")
            raise ValueError(f"筛选列配置错误：{str(e)}")
        except Exception as e:
            logger.error(f"解析筛选列失败：{str(e)}")
            raise ValueError(f"解析筛选列失败：{str(e)}")

    def parse_keywords(self, text):
        if not text.strip():
            return []
        try:
            text = text.replace('，', ',')
            keywords = re.findall(r"'([^']*)'|\"([^\"]*)\",?", text)
            if keywords:
                keywords = [kw[0] or kw[1] for kw in keywords]
            else:
                keywords = [kw.strip().strip("'").strip('"') for kw in text.split(',')]
                keywords = [kw for kw in keywords if kw]
            if not keywords:
                logger.warning(f"关键词解析失败，输入格式无效：{text[:50]}...")
                return []
            return keywords
        except Exception as e:
            logger.error(f"关键词解析错误：{str(e)}")
            return []

    def save_keywords(self):
        if not self.include_keyword_input.text().strip() and not self.exclude_keyword_input.text().strip():
            QMessageBox.warning(self, "提示", "正向和反向关键词均为空，无数据可保存！")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "保存关键词", "", "JSON Files (*.json)")
        if not file_path:
            return
        try:
            positive_keywords = self.parse_keywords(self.include_keyword_input.text())
            negative_keywords = self.parse_keywords(self.exclude_keyword_input.text())
            if not positive_keywords and not negative_keywords:
                raise ValueError("解析后的正向和反向关键词均为空，请检查输入格式！")
            keywords = {
                "positive_keywords": positive_keywords,
                "negative_keywords": negative_keywords
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(keywords, f, indent=2, ensure_ascii=False)
            self.include_keyword_input.setText(", ".join(positive_keywords))
            self.exclude_keyword_input.setText(", ".join(negative_keywords))
            QMessageBox.information(self, "成功", f"关键词保存至：{file_path}")
        except PermissionError as e:
            logger.error(f"保存关键词失败：{str(e)}")
            QMessageBox.critical(self, "错误", f"无法保存文件：{str(e)}")
        except Exception as e:
            logger.error(f"保存关键词失败：{str(e)}")
            QMessageBox.critical(self, "错误", f"保存关键词失败：{str(e)}")

    def load_keywords(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "加载关键词", "", "JSON Files (*.json)")
        if not file_path:
            return
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                keywords = json.load(f)
            positive_keywords = keywords.get("positive_keywords", [])
            negative_keywords = keywords.get("negative_keywords", [])
            if not isinstance(positive_keywords, list) or not isinstance(negative_keywords, list):
                raise ValueError("关键词文件格式错误")
            positive_text = ", ".join(positive_keywords)
            negative_text = ", ".join(negative_keywords)
            self.include_keyword_input.setText(positive_text)
            self.exclude_keyword_input.setText(negative_text)
            QMessageBox.information(self, "成功", "关键词加载成功！")
            self.save_keywords_button.setEnabled(True)
        except json.JSONDecodeError as e:
            logger.error(f"加载关键词失败：无效 JSON 格式，{str(e)}")
            QMessageBox.critical(self, "错误", f"加载关键词失败：无效 JSON 格式")
        except ValueError as e:
            logger.error(f"加载关键词失败：{str(e)}")
            QMessageBox.critical(self, "错误", f"加载关键词失败：{str(e)}")
        except Exception as e:
            logger.error(f"加载关键词失败：{str(e)}")
            QMessageBox.critical(self, "错误", f"加载关键词失败：{str(e)}")

    def tokenize(self, text):
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        if self.language == "中文":
            tokens = jieba.cut(text)
            filtered_tokens = [
                t for t in tokens
                if self.min_token_length <= len(t) <= self.max_token_length
                and re.match(r'^[\u4e00-\u9fa5]+$', t)
                and t not in CHINESE_STOP_WORDS
            ]
            return ' '.join(filtered_tokens)
        return text.strip()

    def extract_keywords_locally(self):
        if self.df is None:
            QMessageBox.warning(self, "错误", "请先导入Excel文件！")
            return
        if self.processing:
            QMessageBox.warning(self, "提示", "正在处理，请稍候...")
            return
        self.processing = True
        self.extract_local_button.setEnabled(False)
        self.ai_keyword_button.setEnabled(False)
        self.filter_button.setEnabled(False)
        self.save_keywords_button.setEnabled(False)
        self.extract_local_button.setText("提取中...")
        threading.Thread(target=self.local_keyword_extraction, daemon=True).start()

    def local_keyword_extraction(self):
        try:
            filter_columns = self.get_filter_columns()
            if filter_columns is None:
                data = self.df.apply(lambda row: ' '.join(str(x) for x in row if x is not None and str(x).strip()), axis=1).dropna().astype(str).tolist()
            else:
                data = self.df.iloc[:, filter_columns].apply(lambda row: ' '.join(str(x) for x in row if x is not None and str(x).strip()), axis=1).dropna().astype(str).tolist()
            
            if not data:
                raise ValueError("选定列的数据为空或全为无效值，无法提取关键词")
            
            data = [item for item in data if item.strip() and not item.strip().isdigit()]

            if not data:
                raise ValueError("清理后数据为空，请检查输入数据")

            stop_words = 'english' if self.language == "英文" else None
            tokenizer = self.tokenize if self.language == "中文" else None

            vectorizer = TfidfVectorizer(
                max_features=2000,  # 增加到 2000
                stop_words=stop_words,
                ngram_range=(1, 5),
                min_df=1,
                tokenizer=tokenizer
            )

            tfidf_matrix = vectorizer.fit_transform(data)
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.sum(axis=0).A1
            
            filtered_features = []
            filtered_scores = []

            for feature, score in zip(feature_names, tfidf_scores):
                feature_cleaned = feature.strip()
                if not feature_cleaned:
                    continue
                if self.language == "中文":
                    if not re.match(r'^[\u4e00-\u9fa5\s]+$', feature_cleaned):
                        continue
                    char_length = len(feature_cleaned.replace(' ', ''))
                    if char_length < self.min_token_length or char_length > self.max_token_length:
                        continue
                else:
                    if not re.match(r'^[a-zA-Z\s]+$', feature_cleaned):
                        continue
                filtered_features.append(feature_cleaned)
                filtered_scores.append(score)

            if not filtered_features:
                raise ValueError("过滤后未提取到任何有效关键词，请调整设置或检查数据")

            total_score = sum(filtered_scores)
            probabilities = np.array(filtered_scores) / total_score if total_score > 0 else np.ones(len(filtered_scores)) / len(filtered_scores)

            num_keywords = min(2000, len(filtered_features))
            np.random.seed(42)

            sampled_indices = np.random.choice(
                len(filtered_features),
                size=num_keywords,
                replace=False,
                p=probabilities
            )

            self.local_keywords = [filtered_features[i] for i in sampled_indices]
            self.keyword_tfidf_scores = {filtered_features[i]: filtered_scores[i] for i in sampled_indices}
            
            logger.debug(f"抽样提取 {len(self.local_keywords)} 个关键词: {self.local_keywords[:20]}...")

            QApplication.instance().postEvent(self, LocalKeywordExtractCompleteEvent())

        except Exception as e:
            logger.error(f"本地关键词提取失败：{str(e)}")
            QApplication.instance().postEvent(self, LocalKeywordExtractErrorEvent(f"本地关键词提取失败：{str(e)}"))

    def extract_keywords_with_deepseek(self):
        if self.df is None:
            QMessageBox.warning(self, "错误", "请先导入Excel文件！")
            return
        if not self.api_key:
            QMessageBox.warning(self, "错误", "请先在设置中配置 DeepSeek API Key！")
            return
        if self.processing:
            QMessageBox.warning(self, "提示", "正在处理，请稍候...")
            return
        self.processing = True
        self.extract_local_button.setEnabled(False)
        self.ai_keyword_button.setEnabled(False)
        self.filter_button.setEnabled(False)
        self.save_keywords_button.setEnabled(False)
        self.ai_keyword_button.setText("提取中...")
        threading.Thread(target=self.call_deepseek_api, daemon=True).start()

    def call_deepseek_api(self):
        try:
            all_positive_keywords = set()
            all_negative_keywords = set()
            remaining_keywords = set()  # 用于补足的剩余关键词
            client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
            if self.local_keywords:
                batch_size = 100  # 减少到 100，避免 API 输入过长
                keywords_batches = [self.local_keywords[i:i + batch_size] for i in range(0, len(self.local_keywords), batch_size)]
                num_batches = len(keywords_batches)
                pos_per_batch = max(1, self.positive_count // num_batches)
                neg_per_batch = max(1, self.negative_count // num_batches)
                for batch_idx, batch in enumerate(keywords_batches):
                    retry_count = 0
                    max_retries = 2
                    while retry_count <= max_retries:
                        try:
                            if batch_idx == num_batches - 1:
                                pos_count = self.positive_count - len(all_positive_keywords)
                                neg_count = self.negative_count - len(all_negative_keywords)
                            else:
                                pos_count = pos_per_batch
                                neg_count = neg_per_batch
                            prompt_message = (
                                f"{self.prompt}\n"
                                f"请提取正向关键词 {pos_count} 个，反向关键词 {neg_count} 个，必须确保数量足够。\n"
                                f"语言：{self.language}\n"
                                f"正向关键词需与设计、艺术、工程、历史名人故居（如‘李小龙故居’）、文化遗产相关，反向关键词需与日常活动、美食、旅游无关。\n"
                                f"以下是一些关键词，请基于这些关键词提取正向和反向关键词（以 JSON 格式返回，包含 'positive_keywords' 和 'negative_keywords' 两个字段）：\n{batch}"
                            )
                            response = client.chat.completions.create(
                                model="deepseek-chat",
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant"},
                                    {"role": "user", "content": prompt_message},
                                ],
                                stream=False,
                                timeout=self.timeout
                            )
                            response_content = response.choices[0].message.content
                            json_match = re.search(r'```json\n([\s\S]*?)\n```', response_content)
                            json_str = json_match.group(1) if json_match else response_content.strip()
                            result = json.loads(json_str)
                            pos_keywords = [kw.replace(" ", "") for kw in result.get("positive_keywords", [])]
                            neg_keywords = [kw.replace(" ", "") for kw in result.get("negative_keywords", [])]
                            all_positive_keywords.update(pos_keywords)
                            all_negative_keywords.update(neg_keywords)
                            # 将未分类的关键词存入 remaining_keywords
                            batch_set = set(batch)
                            remaining_keywords.update(batch_set - set(pos_keywords) - set(neg_keywords))
                            time.sleep(1)
                            break
                        except (json.JSONDecodeError, Exception) as e:
                            retry_count += 1
                            if retry_count > max_retries:
                                logger.error(f"DeepSeek API 处理失败 (批次 {batch_idx+1}/{num_batches})：{str(e)}")
                                break  # 跳出重试，进入补足逻辑
                            time.sleep(5 * (2 ** retry_count))
                # 去重后补足正向关键词
                all_positive_keywords = list(all_positive_keywords)
                while len(all_positive_keywords) < self.positive_count and remaining_keywords:
                    kw = remaining_keywords.pop()
                    if any(design_kw in kw for design_kw in ['设计', '艺术', '工程', '故居', '文化', '非遗']):
                        all_positive_keywords.append(kw)
                    else:
                        all_negative_keywords.append(kw)
                # 去重后补足反向关键词
                all_negative_keywords = list(all_negative_keywords)
                while len(all_negative_keywords) < self.negative_count and remaining_keywords:
                    kw = remaining_keywords.pop()
                    all_negative_keywords.append(kw)
                # 确保数量
                all_positive_keywords = all_positive_keywords[:self.positive_count]
                all_negative_keywords = all_negative_keywords[:self.negative_count]
            else:
                filter_columns = self.get_filter_columns()
                if filter_columns is None:
                    data = self.df.apply(lambda row: ' '.join(str(x) for x in row if x is not None and str(x).strip()), axis=1).dropna().astype(str).tolist()
                else:
                    data = self.df.iloc[:, filter_columns].apply(lambda row: ' '.join(str(x) for x in row if x is not None and str(x).strip()), axis=1).dropna().astype(str).tolist()
                batch_size = self.batch_size
                num_batches = (len(data) + batch_size - 1) // batch_size
                batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
                pos_per_batch = max(1, self.positive_count // num_batches)
                neg_per_batch = max(1, self.negative_count // num_batches)

                def process_batch(batch, batch_idx):
                    retry_count = 0
                    max_retries = 2
                    while retry_count <= max_retries:
                        try:
                            text = ' '.join(batch)[:8000]
                            if batch_idx == num_batches - 1:
                                pos_count = self.positive_count - len(all_positive_keywords)
                                neg_count = self.negative_count - len(all_negative_keywords)
                            else:
                                pos_count = pos_per_batch
                                neg_count = neg_per_batch
                            prompt_message = (
                                f"{self.prompt}\n"
                                f"请提取正向关键词 {pos_count} 个，反向关键词 {neg_count} 个，必须确保数量足够。\n"
                                f"语言：{self.language}\n"
                                f"正向关键词需与设计、艺术、工程、历史名人故居（如‘李小龙故居’）、文化遗产相关，反向关键词需与日常活动、美食、旅游无关。\n"
                                f"以下是文本内容，请从中提取正向和反向关键词（以 JSON 格式返回，包含 'positive_keywords' 和 'negative_keywords' 两个字段）：\n{text}"
                            )
                            response = client.chat.completions.create(
                                model="deepseek-chat",
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant"},
                                    {"role": "user", "content": prompt_message},
                                ],
                                stream=False,
                                timeout=self.timeout
                            )
                            response_content = response.choices[0].message.content
                            json_match = re.search(r'```json\n([\s\S]*?)\n```', response_content)
                            json_str = json_match.group(1) if json_match else response_content.strip()
                            result = json.loads(json_str)
                            pos_keywords = [kw.replace(" ", "") for kw in result.get("positive_keywords", [])]
                            neg_keywords = [kw.replace(" ", "") for kw in result.get("negative_keywords", [])]
                            return pos_keywords, neg_keywords
                        except (json.JSONDecodeError, Exception) as e:
                            retry_count += 1
                            if retry_count > max_retries:
                                logger.error(f"DeepSeek API 处理失败 (批次 {batch_idx+1}/{num_batches})：{str(e)}")
                                return [], []  # 返回空列表，继续处理其他批次
                            time.sleep(5 * (2 ** retry_count))

                with ThreadPoolExecutor(max_workers=self.concurrent_batches) as executor:
                    results = list(executor.map(lambda x: process_batch(x[1], x[0]), enumerate(batches)))
                for pos_keywords, neg_keywords in results:
                    all_positive_keywords.update(pos_keywords)
                    all_negative_keywords.update(neg_keywords)
                # 原始数据模式不补足，直接截取
                all_positive_keywords = list(all_positive_keywords)[:self.positive_count]
                all_negative_keywords = list(all_negative_keywords)[:self.negative_count]

            if not all_positive_keywords or not all_negative_keywords:
                raise ValueError("AI 关键词提取失败，未能生成足够数量的正向或反向关键词")

            positive_text = ", ".join(all_positive_keywords)
            negative_text = ", ".join(all_negative_keywords)
            self.include_keyword_input.setText(positive_text)
            self.exclude_keyword_input.setText(negative_text)

            # 默认保存 JSON
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            default_save_path = os.path.join(os.path.dirname(sys.argv[0]), f"keywords_{timestamp}.json")
            try:
                keywords = {
                    "positive_keywords": all_positive_keywords,
                    "negative_keywords": all_negative_keywords
                }
                with open(default_save_path, 'w', encoding='utf-8') as f:
                    json.dump(keywords, f, indent=2, ensure_ascii=False)
                logger.info(f"关键词自动保存至：{default_save_path}")
            except Exception as e:
                logger.error(f"自动保存关键词失败：{str(e)}")

            QApplication.instance().postEvent(self, KeywordExtractCompleteEvent())

        except Exception as e:
            logger.error(f"DeepSeek API 处理失败：{str(e)}")
            QApplication.instance().postEvent(self, KeywordExtractErrorEvent(f"DeepSeek API 处理失败：{str(e)}，已切换到默认关键词"))
            self.fallback_to_default_keywords()

    def fallback_to_default_keywords(self):
        positive_text = ", ".join(['design', 'art', 'industrial', 'engineering', 'interaction'])
        negative_text = ", ".join(['biology', 'chemistry', 'medicine', 'health', 'physics'])
        self.include_keyword_input.setText(positive_text)
        self.exclude_keyword_input.setText(negative_text)

    def start_filtering(self):
        if self.df is None:
            QMessageBox.warning(self, "错误", "请先导入Excel文件！")
            return
        if self.processing:
            QMessageBox.warning(self, "提示", "正在处理，请稍候...")
            return
        self.processing = True
        self.extract_local_button.setEnabled(False)
        self.ai_keyword_button.setEnabled(False)
        self.filter_button.setEnabled(False)
        self.save_keywords_button.setEnabled(False)
        self.filter_button.setText("筛选中...")
        threading.Thread(target=self.filter_data, daemon=True).start()

    def filter_data(self):
        try:
            include_keywords = self.parse_keywords(self.include_keyword_input.text())
            exclude_keywords = self.parse_keywords(self.exclude_keyword_input.text())
            invalid_include = [kw for kw in include_keywords if not kw.strip()]
            invalid_exclude = [kw for kw in exclude_keywords if not kw.strip()]
            if invalid_include or invalid_exclude:
                QApplication.instance().postEvent(self, FilterErrorEvent("关键词中包含空值或无效字符，请检查输入格式"))
                return
            data_to_filter = self.df if self.filtered_df is None else self.filtered_df
            if not include_keywords and not exclude_keywords:
                self.filtered_df = data_to_filter.copy()
                QApplication.instance().postEvent(self, FilterCompleteEvent(len(self.filtered_df)))
                return
            filter_columns = self.get_filter_columns()
            include_mask = pd.Series(True, index=data_to_filter.index)
            exclude_mask = pd.Series(False, index=data_to_filter.index)
            if include_keywords:
                # 模糊匹配：忽略空格
                include_pattern = '|'.join(r'\s*'.join(re.escape(kw)) for kw in include_keywords)
                if filter_columns is None:
                    temp_col = data_to_filter.apply(lambda row: ' '.join(str(x) for x in row if x is not None and str(x).strip()), axis=1)
                    include_mask = temp_col.str.contains(include_pattern, na=False, regex=True)
                else:
                    temp_col = data_to_filter.iloc[:, filter_columns].apply(lambda row: ' '.join(str(x) for x in row if x is not None and str(x).strip()), axis=1)
                    include_mask = temp_col.str.contains(include_pattern, na=False, regex=True)
            if exclude_keywords:
                exclude_pattern = '|'.join(r'\s*'.join(re.escape(kw)) for kw in exclude_keywords)
                if filter_columns is None:
                    temp_col = data_to_filter.apply(lambda row: ' '.join(str(x) for x in row if x is not None and str(x).strip()), axis=1)
                    exclude_mask = temp_col.str.contains(exclude_pattern, na=False, regex=True)
                else:
                    temp_col = data_to_filter.iloc[:, filter_columns].apply(lambda row: ' '.join(str(x) for x in row if x is not None and str(x).strip()), axis=1)
                    exclude_mask = temp_col.str.contains(exclude_pattern, na=False, regex=True)
            final_mask = include_mask & ~exclude_mask
            self.filtered_df = data_to_filter[final_mask]
            QApplication.instance().postEvent(self, FilterCompleteEvent(len(self.filtered_df)))
        except Exception as e:
            logger.error(f"筛选失败：{str(e)}")
            QApplication.instance().postEvent(self, FilterErrorEvent(str(e)))

    def on_filter_error(self, error_msg):
        QMessageBox.critical(self, "错误", f"筛选失败：{error_msg}")
        self.reset_filter_ui()

    def on_filter_finished(self, num_records):
        if num_records == 0:
            QMessageBox.information(self, "提示", "筛选完成，未找到符合条件的记录！")
        else:
            QMessageBox.information(self, "提示", f"筛选完成，共找到 {num_records} 条记录！")
        self.reset_filter_ui()

    def reset_filter_ui(self):
        self.extract_local_button.setText("本地关键词提取")
        self.ai_keyword_button.setText("AI关键词")
        self.filter_button.setText("筛选")
        self.extract_local_button.setEnabled(bool(self.df is not None))
        self.ai_keyword_button.setEnabled(bool(self.api_key))
        self.filter_button.setEnabled(True)
        self.save_keywords_button.setEnabled(True)
        self.processing = False

    def customEvent(self, event):
        if isinstance(event, FilterCompleteEvent):
            self.on_filter_finished(event.num_records)
        elif isinstance(event, FilterErrorEvent):
            self.on_filter_error(event.error_msg)
        elif isinstance(event, LocalKeywordExtractCompleteEvent):
            QMessageBox.information(self, "提示", f"本地关键词提取完成，共提取 {len(self.local_keywords)} 个关键词！")
            self.extract_local_button.setText("本地关键词提取")
            self.ai_keyword_button.setText("AI关键词")
            self.filter_button.setText("筛选")
            self.extract_local_button.setEnabled(True)
            self.ai_keyword_button.setEnabled(bool(self.api_key))
            self.filter_button.setEnabled(True)
            self.save_keywords_button.setEnabled(True)
        elif isinstance(event, LocalKeywordExtractErrorEvent):
            QMessageBox.critical(self, "错误", event.error_msg)
            self.extract_local_button.setText("本地关键词提取")
            self.ai_keyword_button.setText("AI关键词")
            self.filter_button.setText("筛选")
            self.extract_local_button.setEnabled(bool(self.df is not None))
            self.ai_keyword_button.setEnabled(bool(self.api_key))
            self.filter_button.setEnabled(True)
            self.save_keywords_button.setEnabled(True)
        elif isinstance(event, KeywordExtractCompleteEvent):
            QMessageBox.information(self, "提示", "AI 关键词分类完成！")
            self.extract_local_button.setText("本地关键词提取")
            self.ai_keyword_button.setText("AI关键词")
            self.filter_button.setText("筛选")
            self.extract_local_button.setEnabled(bool(self.df is not None))
            self.ai_keyword_button.setEnabled(bool(self.api_key))
            self.filter_button.setEnabled(True)
            self.save_keywords_button.setEnabled(True)
        elif isinstance(event, KeywordExtractErrorEvent):
            QMessageBox.critical(self, "错误", event.error_msg)
            self.extract_local_button.setText("本地关键词提取")
            self.ai_keyword_button.setText("AI关键词")
            self.filter_button.setText("筛选")
            self.extract_local_button.setEnabled(bool(self.df is not None))
            self.ai_keyword_button.setEnabled(bool(self.api_key))
            self.filter_button.setEnabled(True)
            self.save_keywords_button.setEnabled(True)
        self.processing = False

    def export_file(self):
        if self.filtered_df is None:
            QMessageBox.warning(self, "错误", "请先进行筛选！")
            return
        if self.import_file_path is None:
            QMessageBox.warning(self, "错误", "请先导入文件！")
            return
        file_dir = os.path.dirname(self.import_file_path)
        file_name = os.path.basename(self.import_file_path)
        file_name_without_ext, ext = os.path.splitext(file_name)
        export_file_path = os.path.join(file_dir, f"{file_name_without_ext}筛选{ext}")
        try:
            if os.path.exists(export_file_path):
                try:
                    with open(export_file_path, 'a'):
                        pass
                except PermissionError:
                    raise PermissionError(f"无法写入文件：{export_file_path}\n请关闭文件后重试！")
            self.filtered_df.to_excel(export_file_path, index=False)
            QMessageBox.information(self, "成功", f"导出成功，文件保存至：{export_file_path}")
        except PermissionError as e:
            logger.error(f"导出失败：{str(e)}")
            QMessageBox.critical(self, "错误", str(e))
        except Exception as e:
            logger.error(f"导出失败：{str(e)}")
            QMessageBox.critical(self, "错误", f"导出失败：{str(e)}")

class FilterCompleteEvent(QEvent):
    def __init__(self, num_records):
        super().__init__(QEvent.Type(QEvent.Type.User + 1))
        self.num_records = num_records

class FilterErrorEvent(QEvent):
    def __init__(self, error_msg):
        super().__init__(QEvent.Type(QEvent.Type.User + 2))
        self.error_msg = error_msg

class LocalKeywordExtractCompleteEvent(QEvent):
    def __init__(self):
        super().__init__(QEvent.Type(QEvent.Type.User + 3))

class LocalKeywordExtractErrorEvent(QEvent):
    def __init__(self, error_msg):
        super().__init__(QEvent.Type(QEvent.Type.User + 4))
        self.error_msg = error_msg

class KeywordExtractCompleteEvent(QEvent):
    def __init__(self):
        super().__init__(QEvent.Type(QEvent.Type.User + 5))

class KeywordExtractErrorEvent(QEvent):
    def __init__(self, error_msg):
        super().__init__(QEvent.Type(QEvent.Type.User + 6))
        self.error_msg = error_msg

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DesignFilterTool()
    window.show()
    sys.exit(app.exec())