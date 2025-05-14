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

# 设置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 尝试导入 langdetect 和 jieba
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect 未安装，将使用默认语言 'English'。请安装：pip install langdetect")

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logger.warning("jieba 未安装，中文分词将不可用。请安装：pip install jieba")


class ConnectionTestCompleteEvent(QEvent):
    def __init__(self):
        super().__init__(QEvent.Type(QEvent.Type.User + 5))

class ConnectionTestErrorEvent(QEvent):
    def __init__(self, error_msg):
        super().__init__(QEvent.Type(QEvent.Type.User + 6))
        self.error_msg = error_msg

# 用于信号通信
class FilterSignals(QObject):
    error = pyqtSignal(str)     # 错误信息
    finished = pyqtSignal(int)  # 筛选完成

class SettingsDialog(QDialog):
    def __init__(self, parent=None, available_columns=None):
        super().__init__(parent)
        self.available_columns = available_columns or []
        self.setWindowTitle("设置")
        self.setGeometry(400, 400, 400, 500)

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
        self.language_input.addItems(["English", "Chinese", "Other"])
        if not LANGDETECT_AVAILABLE:
            self.language_input.setEnabled(False)
            self.language_input.setToolTip("langdetect 未安装，无法自动检测语言，默认使用 English")
        self.layout.addRow(self.language_label, self.language_input)

        self.max_features_label = QLabel("最大关键词数量:")
        self.max_features_input = QLineEdit()
        self.max_features_input.setPlaceholderText("请输入最大关键词数量（默认 1000，越大提取关键词越多）")
        self.max_features_input.setValidator(validator)
        self.max_features_input.setToolTip("控制本地关键词提取的词汇数量，值越大提取的关键词越多，但可能包含更多噪声")
        self.layout.addRow(self.max_features_label, self.max_features_input)

        self.filter_columns_label = QLabel("筛选列（列名）：")
        self.filter_columns_input = QLineEdit()
        placeholder = "输入列名，英文逗号分隔，例如：Title,Abstract（留空使用所有列，无需引号）"
        if self.available_columns:
            placeholder += f"\n可用列名：{', '.join(self.available_columns)}"
        self.filter_columns_input.setPlaceholderText(placeholder)
        self.filter_columns_input.setToolTip("请输入 Excel 文件中的列名，多个列名用英文逗号分隔（,），无需引号，留空使用所有列；影响本地关键词提取、AI 关键词提取和筛选过程")
        self.layout.addRow(self.filter_columns_label, self.filter_columns_input)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept_with_validation)
        self.buttons.rejected.connect(self.reject)
        self.layout.addRow(self.buttons)

        self.load_settings()
        self.setLayout(self.layout)
        self.msg_box = None

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
                self.language_input.setCurrentText(settings.get("language", "English"))
                self.max_features_input.setText(str(settings.get("max_features", "")))
                self.filter_columns_input.setText(settings.get("filter_columns", ""))
            except Exception as e:
                logger.error(f"加载设置失败：{str(e)}")
                QMessageBox.warning(self, "错误", f"加载设置失败：{str(e)}")

    def validate_filter_columns(self, text, columns=None):
        if not text.strip():
            return True, ""
        if columns is None:
            return True, ""  # 延迟验证
        # 支持中文逗号和英文逗号
        text = text.replace('，', ',')
        column_names = [col.strip().strip("'").strip('"') for col in text.split(',')]
        # 检查是否使用了中文逗号
        if '，' in text and not ',' in text:
            return False, "请使用英文逗号（,）分隔列名，而不是中文逗号（，）"
        # 检测数字输入
        if any(col.isdigit() for col in column_names):
            return False, "请输入列名而非列号（例如：Title,Abstract，无需引号）"
        # 忽略大小写匹配
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

        max_features = self.max_features_input.text().strip()
        max_features_value = int(max_features) if max_features else 1000
        max_features_value = max(100, min(5000, max_features_value))

        settings = {
            "api_key": self.api_key_input.text().strip(),
            "prompt": self.prompt_input.toPlainText().strip(),
            "positive_count": int(self.positive_count_input.text().strip()) if self.positive_count_input.text().strip() else 15,
            "negative_count": int(self.negative_count_input.text().strip()) if self.negative_count_input.text().strip() else 50,
            "timeout": timeout_value,
            "language": self.language_input.currentText(),
            "max_features": max_features_value,
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
        max_features = self.max_features_input.text().strip()
        settings = {
            "api_key": self.api_key_input.text().strip(),
            "prompt": self.prompt_input.toPlainText().strip() or "请帮我提炼关于设计、艺术设计、交互设计、工程设计、工业设计相关的正向关键词，以及无关的生物、化学、人类学等反向关键词",
            "positive_count": int(positive_count) if positive_count else 15,
            "negative_count": int(negative_count) if negative_count else 50,
            "timeout": int(timeout) if timeout else 60,
            "language": self.language_input.currentText(),
            "max_features": int(max_features) if max_features else 1000,
            "filter_columns": self.filter_columns_input.text().strip()
        }
        self.save_settings()
        return settings

class DesignFilterTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("筛选器")
        self.setGeometry(300, 300, 600, 300)

        self.setWindowIcon(QIcon('filter.ico'))

        self.settings_action = QPushButton()
        self.settings_action.setIcon(QIcon('setting.ico'))
        self.settings_action.setToolTip("设置 DeepSeek API")
        self.settings_action.setFixedSize(30, 30)
        self.settings_action.clicked.connect(self.show_settings_dialog)
        self.setMenuWidget(self.settings_action)

        self.api_key = ""
        self.prompt = ""
        self.positive_count = 15
        self.negative_count = 50
        self.timeout = 60
        self.language = "English"
        self.max_features = 1000
        self.filter_columns = ""

        self.load_settings()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.include_keyword_layout = QHBoxLayout()
        self.include_keyword_label = QLabel("正向关键词：")
        self.include_keyword_input = QLineEdit()
        self.include_keyword_input.setPlaceholderText("输入或等待 AI 提取，例如 'design', 'industrial'")
        self.include_keyword_layout.addWidget(self.include_keyword_label)
        self.include_keyword_layout.addWidget(self.include_keyword_input)
        self.layout.addLayout(self.include_keyword_layout)

        self.exclude_keyword_layout = QHBoxLayout()
        self.exclude_keyword_label = QLabel("反向关键词：")
        self.exclude_keyword_input = QLineEdit()
        self.exclude_keyword_input.setPlaceholderText("输入或等待 AI 提取，例如 'biology', 'medicine'")
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

        # 连接信号
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
                self.negative_count = settings.get("negative_count", 50)
                self.timeout = settings.get("timeout", 60)
                self.language = settings.get("language", "English")
                self.max_features = settings.get("max_features", 1000)
                self.filter_columns = settings.get("filter_columns", "")
                self.timeout = max(10, min(300, self.timeout))
                self.max_features = max(100, min(5000, self.max_features))
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
            self.max_features = settings["max_features"]
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
            self.ai_keyword_button.setEnabled(bool(self.api_key))  # 导入后即可使用 AI 关键词
            self.filter_button.setEnabled(True)
            self.save_keywords_button.setEnabled(True)

            column_info = f"列名（共 {len(self.column_names)} 列）：\n" + ", ".join(f"{i+1}: {col}" for i, col in enumerate(self.column_names))
            QMessageBox.information(self, "提示", f"导入成功，共 {len(self.df)} 条数据！\n\n{column_info}")
        except Exception as e:
            logger.error(f"无法读取文件：{str(e)}")
            QMessageBox.critical(self, "错误", f"无法读取文件：{str(e)}（请确保文件是有效的 Excel 格式）")
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
            # 支持中文逗号和英文逗号
            filter_columns = self.filter_columns.replace('，', ',')
            column_names = [col.strip().strip("'").strip('"') for col in filter_columns.split(',')]
            columns = []
            column_names_lower = [col.lower() for col in self.column_names]
            for col in column_names:
                if col.lower() not in column_names_lower:
                    raise ValueError(f"无效列名：{col}（可用列名：{', '.join(self.column_names)}）")
                # 找到原始列名对应的索引
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
            # 提取以单引号或双引号包裹的关键词
            keywords = re.findall(r"'([^']*)'|\"([^\"]*)\",?", text)
            keywords = [kw[0] or kw[1] for kw in keywords]
            if not keywords:
                # 如果正则匹配失败，尝试按逗号分隔并清理
                keywords = [kw.strip().strip("'").strip('"') for kw in text.split(',')]
                keywords = [kw for kw in keywords if kw]  # 移除空字符串
            if not keywords:
                logger.warning(f"关键词解析失败，输入格式无效：{text[:50]}...")
                return []
            return keywords
        except Exception as e:
            logger.error(f"关键词解析错误：{str(e)}")
            return []

    def save_keywords(self):
        if not self.include_keyword_input.text().strip() and not self.exclude_keyword_input.text().strip():
            QMessageBox.warning(self, "提示", "正向和反向关键词均为空，无需保存！")
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
            QMessageBox.information(self, "成功", f"关键词保存至：{file_path}")
        except PermissionError as e:
            logger.error(f"保存关键词失败：{str(e)}")
            QMessageBox.critical(self, "错误", f"无法保存文件：{str(e)}（请检查文件权限）")
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
                raise ValueError("关键词文件格式错误：positive_keywords 和 negative_keywords 必须是列表")
            positive_text = ", ".join(f"'{kw}'" for kw in positive_keywords)
            negative_text = ", ".join(f"'{kw}'" for kw in negative_keywords)
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
        if self.language == "Chinese" and JIEBA_AVAILABLE:
            return ' '.join(jieba.cut(text))
        return text

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
                data = self.df.apply(lambda row: ' '.join(row), axis=1).dropna().astype(str).tolist()
            else:
                data = self.df.iloc[:, filter_columns].apply(lambda row: ' '.join(row), axis=1).dropna().astype(str).tolist()

            if not data:
                raise ValueError("选定列的数据为空或全为无效值，无法提取关键词")

            stop_words = 'english' if self.language == "English" else None
            tokenizer = self.tokenize if self.language == "Chinese" and JIEBA_AVAILABLE else None
            vectorizer = TfidfVectorizer(max_features=None, stop_words=stop_words, ngram_range=(1, 2), min_df=5, tokenizer=tokenizer)
            tfidf_matrix = vectorizer.fit_transform(data)
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.sum(axis=0).A1

            if len(feature_names) == 0:
                raise ValueError("未提取到任何关键词，可能数据过于稀疏或无效")

            total_score = sum(tfidf_scores)
            probabilities = tfidf_scores / total_score if total_score > 0 else np.ones(len(tfidf_scores)) / len(tfidf_scores)

            num_keywords = min(self.max_features, len(feature_names))
            if num_keywords == 0:
                raise ValueError("词汇数量不足，无法进行抽样")
            np.random.seed(42)
            sampled_indices = np.random.choice(
                len(feature_names),
                size=num_keywords,
                replace=False,
                p=probabilities
            )
            self.local_keywords = [feature_names[i] for i in sampled_indices]
            self.keyword_tfidf_scores = {feature_names[i]: tfidf_scores[i] for i in sampled_indices}

            logger.debug(f"抽样提取 {len(self.local_keywords)} 个关键词: {self.local_keywords[:20]}...")
            logger.debug(f"TF-IDF 分数 (前 20): {dict(list(self.keyword_tfidf_scores.items())[:20])}")

            # 自动保存到 Excel
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            excel_path = os.path.join(os.path.dirname(sys.argv[0]), f"local_keywords_{timestamp}.xlsx")
            keywords_df = pd.DataFrame({
                "Keyword": self.local_keywords,
                "TF-IDF Score": [self.keyword_tfidf_scores[kw] for kw in self.local_keywords],
                "Timestamp": [time.strftime("%Y-%m-%d %H:%M:%S")] * len(self.local_keywords)
            })
            keywords_df.to_excel(excel_path, index=False)
            logger.info(f"本地关键词已保存至：{excel_path}")

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

            client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

            # 如果有本地关键词，则基于本地关键词进行分类
            if self.local_keywords:
                batch_size = 200
                keywords_batches = [self.local_keywords[i:i + batch_size] for i in range(0, len(self.local_keywords), batch_size)]

                for batch_idx, batch in enumerate(keywords_batches):
                    retry_count = 0
                    max_retries = 2
                    while retry_count <= max_retries:
                        try:
                            prompt_message = (
                                f"{self.prompt}\n"
                                f"请提取正向关键词 {self.positive_count} 个，反向关键词 {self.negative_count} 个，确保数量足够。\n"
                                f"语言：{self.language}\n"
                                f"以下是一些关键词，请基于这些关键词提取正向和反向关键词（以 JSON 格式返回，包含 'positive_keywords' 和 'negative_keywords' 两个字段）：\n{batch}"
                            )
                            logger.debug(f"发送 DeepSeek 请求 (批次 {batch_idx + 1}/{len(keywords_batches)}): {prompt_message[:500]}...")

                            response = client.chat.completions.create(
                                model="deepseek-chat",
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant"},
                                    {"role": "user", "content": prompt_message},
                                ],
                                stream=False,
                                timeout=self.timeout
                            )

                            if not hasattr(response, 'choices') or not response.choices:
                                raise ValueError("DeepSeek API 返回空响应，请检查 API Key 或网络连接")

                            response_content = response.choices[0].message.content
                            if not response_content:
                                raise ValueError("DeepSeek API 返回内容为空，请检查 API Key 或请求内容")

                            logger.debug(f"DeepSeek 响应 (批次 {batch_idx + 1}/{len(keywords_batches)}): {response_content[:500]}...")

                            json_match = re.search(r'```json\n([\s\S]*?)\n```', response_content)
                            if json_match:
                                json_str = json_match.group(1)
                            else:
                                json_str = response_content.strip()

                            result = json.loads(json_str)
                            if "positive_keywords" not in result or "negative_keywords" not in result:
                                raise ValueError("DeepSeek 响应缺少必要字段：positive_keywords 或 negative_keywords")

                            positive_keywords = set(result.get("positive_keywords", []))
                            negative_keywords = set(result.get("negative_keywords", []))
                            all_positive_keywords.update(positive_keywords)
                            all_negative_keywords.update(negative_keywords)
                            break
                        except json.JSONDecodeError as e:
                            retry_count += 1
                            logger.warning(f"JSON 解析错误 (批次 {batch_idx + 1}/{len(keywords_batches)}，第 {retry_count} 次重试)：{str(e)}")
                            if retry_count > max_retries:
                                raise ValueError(f"DeepSeek 响应格式错误：{str(e)}")
                            time.sleep(5)
                        except Exception as e:
                            retry_count += 1
                            logger.warning(f"DeepSeek API 调用失败 (批次 {batch_idx + 1}/{len(keywords_batches)}，第 {retry_count} 次重试)：{str(e)}")
                            if retry_count > max_retries:
                                raise Exception(f"DeepSeek API 调用失败：{str(e)}")
                            time.sleep(5)

                remaining_keywords = [kw for kw in self.local_keywords if kw not in all_positive_keywords and kw not in all_negative_keywords]
                design_related = ['design', 'art', 'arts', 'interaction', 'interactive', 'engineering', 'industrial', 'graphics', 'visual']
                while len(all_positive_keywords) < self.positive_count and remaining_keywords:
                    next_keyword = remaining_keywords.pop(0)
                    if any(design_kw in next_keyword for design_kw in design_related):
                        all_positive_keywords.add(next_keyword)

                non_design_related = ['biology', 'chemistry', 'medicine', 'health', 'physics', 'psychology']
                while len(all_negative_keywords) < self.negative_count and remaining_keywords:
                    next_keyword = remaining_keywords.pop(0)
                    if any(non_design_kw in next_keyword for non_design_kw in non_design_related):
                        all_negative_keywords.add(next_keyword)

            # 如果没有本地关键词，直接从原始数据提取
            else:
                filter_columns = self.get_filter_columns()
                if filter_columns is None:
                    data = self.df.apply(lambda row: ' '.join(row), axis=1).dropna().astype(str).tolist()
                else:
                    data = self.df.iloc[:, filter_columns].apply(lambda row: ' '.join(row), axis=1).dropna().astype(str).tolist()

                if not data:
                    raise ValueError("选定列的数据为空或全为无效值，无法提取关键词")

                # 将数据拼接为单一文本
                text = ' '.join(data)
                retry_count = 0
                max_retries = 2
                while retry_count <= max_retries:
                    try:
                        prompt_message = (
                            f"{self.prompt}\n"
                            f"请提取正向关键词 {self.positive_count} 个，反向关键词 {self.negative_count} 个，确保数量足够。\n"
                            f"语言：{self.language}\n"
                            f"以下是文本内容，请从中提取正向和反向关键词（以 JSON 格式返回，包含 'positive_keywords' 和 'negative_keywords' 两个字段）：\n{text[:10000]}"  # 限制文本长度
                        )
                        logger.debug(f"发送 DeepSeek 请求 (直接从文本提取): {prompt_message[:500]}...")

                        response = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant"},
                                {"role": "user", "content": prompt_message},
                            ],
                            stream=False,
                            timeout=self.timeout
                        )

                        if not hasattr(response, 'choices') or not response.choices:
                            raise ValueError("DeepSeek API 返回空响应，请检查 API Key 或网络连接")

                        response_content = response.choices[0].message.content
                        if not response_content:
                            raise ValueError("DeepSeek API 返回内容为空，请检查 API Key 或请求内容")

                        logger.debug(f"DeepSeek 响应 (直接从文本提取): {response_content[:500]}...")

                        json_match = re.search(r'```json\n([\s\S]*?)\n```', response_content)
                        if json_match:
                            json_str = json_match.group(1)
                        else:
                            json_str = response_content.strip()

                        result = json.loads(json_str)
                        if "positive_keywords" not in result or "negative_keywords" not in result:
                            raise ValueError("DeepSeek 响应缺少必要字段：positive_keywords 或 negative_keywords")

                        all_positive_keywords = set(result.get("positive_keywords", []))
                        all_negative_keywords = set(result.get("negative_keywords", []))
                        break
                    except json.JSONDecodeError as e:
                        retry_count += 1
                        logger.warning(f"JSON 解析错误 (直接从文本提取，第 {retry_count} 次重试)：{str(e)}")
                        if retry_count > max_retries:
                            raise ValueError(f"DeepSeek 响应格式错误：{str(e)}")
                        time.sleep(5)
                    except Exception as e:
                        retry_count += 1
                        logger.warning(f"DeepSeek API 调用失败 (直接从文本提取，第 {retry_count} 次重试)：{str(e)}")
                        if retry_count > max_retries:
                            raise Exception(f"DeepSeek API 调用失败：{str(e)}")
                        time.sleep(5)

            positive_keywords = list(all_positive_keywords)
            negative_keywords = list(all_negative_keywords)
            positive_text = ", ".join(f"'{kw}'" for kw in positive_keywords)
            negative_text = ", ".join(f"'{kw}'" for kw in negative_keywords)
            self.include_keyword_input.setText(positive_text)
            self.exclude_keyword_input.setText(negative_text)

            QApplication.instance().postEvent(self, KeywordExtractCompleteEvent())
        except Exception as e:
            logger.error(f"DeepSeek API 处理失败：{str(e)}")
            QApplication.instance().postEvent(self, KeywordExtractErrorEvent(f"DeepSeek API 处理失败：{str(e)}，已切换到默认关键词"))
            self.fallback_to_default_keywords()

    def fallback_to_default_keywords(self):
        positive_text = ", ".join(f"'{kw}'" for kw in ['design', 'art', 'industrial', 'engineering', 'interaction'])
        negative_text = ", ".join(f"'{kw}'" for kw in ['biology', 'chemistry', 'medicine', 'health', 'physics'])
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

            # 矢量化处理正向关键词
            if include_keywords:
                include_pattern = '|'.join(map(re.escape, include_keywords))
                if filter_columns is None:
                    temp_col = data_to_filter.apply(lambda row: ' '.join(row), axis=1)
                    include_mask = temp_col.str.contains(include_pattern, na=False, regex=True)
                else:
                    temp_col = data_to_filter.iloc[:, filter_columns].apply(lambda row: ' '.join(row), axis=1)
                    include_mask = temp_col.str.contains(include_pattern, na=False, regex=True)

            # 矢量化处理反向关键词
            if exclude_keywords:
                exclude_pattern = '|'.join(map(re.escape, exclude_keywords))
                if filter_columns is None:
                    temp_col = data_to_filter.apply(lambda row: ' '.join(row), axis=1)
                    exclude_mask = temp_col.str.contains(exclude_pattern, na=False, regex=True)
                else:
                    temp_col = data_to_filter.iloc[:, filter_columns].apply(lambda row: ' '.join(row), axis=1)
                    exclude_mask = temp_col.str.contains(exclude_pattern, na=False, regex=True)

            # 组合条件
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
            # 仅显示提取的总数量，移除“前 20 个关键词”展示
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