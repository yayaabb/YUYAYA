# AI集合用户标签提取 + 旅游广告生成系统

本项目是一个集 **AI 用户标签分析** ＋ **Stable Diffusion 旅游描述图像生成** ＋ **AI文案叠图动画广告设计** 三合一的智能广告工具链。

---

## 总体流程

1. 引入用户基础信息表 + 搜索行为文本
2. 调用 LLaMA3 本地模型，生成标准化的用户旅行偏好标签 JSON
3. 同步 Stable Diffusion + LoRA 样式，根据用户描述生成旅游描述插画
4. 调用 LLaMA3 旅游广告语，与图片合成动态GIF

---

## 方法涉及

### 【A】 AI 用户标签提取 （`user_tag_pipeline_full_cleaned.py`）

* 根据用户搜索行为和个人信息，通过 prompt 调用 LLaMA3 API
* 生成包含 7 个综合类目的 JSON 标签结果
* 具有语义证据判断冲突和标准化处理

### 【B】 Stable Diffusion 图像生成 （`image_generation.py`）

* 调用 Hugging Face FLUX.1-dev + LoRA （Ghibli）
* 根据用户标签 + prompt 生成实际旅游场景图
* 高颜值精致描述，适合礼品/示意广告场景

### 【C】 AI 文案叠图 + 广告GIF动画 （`Slogan_generate.py`）

* 调用 LLaMA3 编写密切、情感化旅游广告语
* 利用 DETR 物体检测，自动避应绘文区域
* 实现多种动效：漲动、渐变、演打机、波动...
* 最终生成动态广告GIF

---

## 输出文件

* `user_tag_outputs_merged.csv`
* `Mobileoutput/PCoutput_xx.png`（个性化图片生成）
* `final_outputs/mobile/pc/mb_ad_xx.gif`（广告动画效果）


