# 词汇挑战 AI

一个面向**国际学生**的 AI 驱动英语词汇挑战游戏，结合**图像识别**与**语音交互**，帮助提升日常英语口语表达能力。

用户将看到现实场景图像（来自 COCO 数据集），并需**大声说出图中物品名称**。系统使用 **OpenAI Whisper** 进行语音识别，结合**模糊匹配与同义词归一化**判断答案正确性。

并且新增了 **YOLOv8 目标检测** ，用户可以实时上传自己的图片，进行词汇学习

---

## 功能特色

- Whisper 语音识别
- COCO 图像词汇挑战
- YOLOv8 实时目标检测，定位图片物体 
- 模糊匹配 + 同义词处理
- 错题记忆与复习模式
- 本地排行榜（score.json）
- 模块化架构，便于拓展

---

## 项目结构

```bash
project-1/
├── main.py                       # Streamlit 项目入口
├── app/
│   ├── audio_handler.py          # Whisper 转录模块
│   ├── config.py                 # 参数配置
│   ├── game_logic.py             # 主游戏逻辑控制
│   ├── image_loader.py           # COCO 数据加载
│   ├── score_board.py            # 分数管理与排行榜
│   ├── utils_text.py             # 文本预处理与模糊匹配
│   └── yolo_detector.py          # YOLOv8 检测模块（新增）
├── coco_dataset/
│   ├── images/val2017/           # 图像文件（需单独下载）
│   └── annotations/instances_val2017.json
├── requirements.txt
├── README.md
└── .gitignore

```

---

##  安装

### 克隆链接与安装库
```bash
git clone https://github.com/yayaabb/YUYAYA
cd vocabulary_challenge_ai
pip install -r requirements.txt
```

---

##  运行APP

```bash
cd project-1
streamlit run main.py
```

使用侧边栏
- 开始新游戏
- 重玩错误
- 查看排行榜
- 重置会话
- 图像识别

