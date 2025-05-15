Design Filter Tool
项目简介
Design Filter Tool 是一个基于 Python 的 GUI 工具，用于从文本数据中提取关键词并进行筛选。程序支持本地关键词提取（基于 TF-IDF）和 AI 关键词提取（通过 DeepSeek API），适用于设计、艺术、工程等领域的关键词分析和数据过滤。

开发者：未知（可补充）
版本：1.0
日期：2025-05-14

功能介绍
1. 本地关键词提取

使用 TF-IDF 算法从文本数据中提取关键词。
支持中文分词（基于 jieba），可设置分词长度（默认 2-5 字符）。
提供 250 个默认中文停用词，用户可自定义停用词。

2. AI 关键词提取

通过 DeepSeek API 提取正向和反向关键词。
正向关键词：与设计、艺术、工程、历史名人故居（如“李小龙故居”）、文化遗产相关。
反向关键词：与日常活动、美食、旅游等无关。
支持分批处理和并发请求，提高效率。

3. 数据筛选

根据正向和反向关键词筛选数据，支持模糊匹配（忽略空格）。
用户可选择筛选列，灵活处理 Excel 数据。

4. 自定义设置

支持设置 DeepSeek API Key、正向/反向关键词数量、超时时间、语言、分词长度、批次大小等。
设置保存在 settings.json，程序启动时自动加载。
可编辑停用词，保存到 stopwords.txt。

5. 数据导入导出

支持导入 Excel 文件（.xlsx, .xls）。
筛选结果可导出为 Excel 文件。

安装步骤
1. 环境要求

Python 3.7 或以上版本。
Windows 系统（其他系统需调整 PyInstaller 打包命令）。

2. 安装依赖包
使用国内清华镜像源加速安装：
pip install openpyxl pandas numpy scikit-learn PyQt6 openai jieba -i https://pypi.tuna.tsinghua.edu.cn/simple

3. 安装打包工具（可选）
如果需要打包成 .exe 文件，安装 PyInstaller：
pip install pyinstaller -i https://pypi.tuna.tsinghua.edu.cn/simple

4. 下载程序文件

确保 design_filter_tool.py、filter.ico 和 setting.ico 文件位于同一目录（推荐 d:/design_filter/）。
filter.ico：主窗口标题和 .exe 文件图标。
setting.ico：设置按钮图标。

使用方法
1. 运行程序
直接运行 Python 脚本：
cd d:/design_filter
python design_filter_tool.py

2. 界面操作

导入数据：点击“导入”按钮，选择 Excel 文件（.xlsx 或 .xls）。
本地关键词提取：点击“本地关键词提取”，从数据中提取关键词。
AI 关键词提取：
确保设置中已配置 DeepSeek API Key。
点击“AI关键词”，调用 DeepSeek API 提取正向和反向关键词。


筛选数据：点击“筛选”，根据正向和反向关键词筛选数据。
保存关键词：点击“保存关键词”，将提取的关键词保存为 .json 文件。
加载关键词：点击“加载关键词”，从 .json 文件加载关键词。
导出结果：点击“导出”，将筛选后的数据保存为 Excel 文件。
设置：
点击设置按钮（右上角），可配置 API Key、正向/反向关键词数量等。
支持编辑停用词，保存后生效。



3. 配置文件

settings.json：存储用户设置，位于程序目录。
stopwords.txt：存储停用词，位于程序目录。

打包成 .exe 文件
1. 打包命令
在 d:/design_filter/ 目录下运行以下命令：
pyinstaller -F -w --name DesignFilterTool --icon=filter.ico --add-data "filter.ico;." --add-data "setting.ico;." design_filter_tool.py

2. 打包说明

-F：打包成单个 .exe 文件。
-w：去掉控制台窗口（GUI 程序适用）。
--name DesignFilterTool：输出文件名为 DesignFilterTool.exe。
--icon=filter.ico：设置 .exe 文件图标。
--add-data "filter.ico;." 和 --add-data "setting.ico;."：打包图标文件。
输出路径：d:/design_filter/dist/DesignFilterTool.exe。

3. 运行 .exe
双击 dist/DesignFilterTool.exe 运行程序，无需 Python 环境。
注意事项

DeepSeek API Key：需自行申请，设置中配置后才能使用 AI 关键词提取。
Excel 文件：确保导入的 Excel 文件格式正确，列名无特殊字符。
网络环境：AI 关键词提取需要联网，建议使用国内镜像源加速依赖安装。
打包问题：若打包失败，可去掉 -F 参数查看详细依赖，或添加 --hidden-import 指定缺失模块。

联系方式
如有问题，请联系开发者（可补充联系方式）。
