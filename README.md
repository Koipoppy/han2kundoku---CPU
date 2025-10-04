# 汉文到训读格式转换模型（Kundoku Converter - CPU）

## 项目用途
本项目旨在构建一个自动化模型，实现从**汉文句子**到**日语训读格式**的精准转换。训读格式是汉文日语化的核心表示形式，每个汉字对应包含「汉字本身、序号标注（如“二”“一”）、假名注音（如“モ”“ツ”）」的三元组结构（示例：`[["声", "", ""], ["声", "", "モ"], ["断", "二", "ツ"]]`）。

项目可应用于：
- 古典汉文文献的日语训读自动化标注
- 日语教育中汉文训读学习辅助工具
- 跨语言（汉-日）古籍数字化处理流程

## 环境配置

### 1. 系统要求
- 操作系统：Windows/macOS/Linux
- 硬件：CPU可运行，GPU（NVIDIA CUDA）可加速训练（推荐）
- 内存：至少8GB（数据量较大时建议16GB以上）

### 2. 依赖安装
首先确保已安装Python 3.8+，然后通过`requirements.txt`安装依赖库：
```bash
# 克隆项目后进入根目录
cd kundoku_converter
# 安装依赖
pip install -r requirements.txt
```

`requirements.txt`包含的核心依赖：
- `torch>=1.10.0`：PyTorch深度学习框架
- `transformers>=4.0.0`：预训练模型支持（可选，用于升级模型）
- `numpy>=1.19.0`：数值计算
- `pandas>=1.3.0`：数据处理（可选）
- `scikit-learn>=0.24.0`：评估指标（可选）


## 数据格式
模型训练和验证依赖特定格式的汉文-训读配对数据，数据需放在`data/raw/`目录下，目录结构如下：
```
data/raw/
├── train/          # 训练集
│   ├── hanwen.txt  # 汉文句子文件
│   └── kundoku.jsonl  # 对应训读格式文件
└── valid/          # 验证集
    ├── hanwen.txt
    └── kundoku.jsonl
```

### 1. 汉文文件（hanwen.txt）
- 格式：**每行一句汉文**，句子编码为UTF-8
- 要求：句子无空格、无标点（或统一预处理后去除标点）
- 示例：
  ```
  声声断人肠
  春眠不觉晓
  床前明月光
  ```

### 2. 训读格式文件（kundoku.jsonl）
- 格式：**JSON Lines（每行一个JSON数组）**，与`hanwen.txt`的句子一一对应
- 核心结构：每个句子对应一个「三元组列表」，三元组为`[汉字, 序号, 假名]`（序号/假名为空时用空字符串`""`）
- 要求：三元组列表长度必须与对应汉文句子的汉字数一致
- 示例（与上述汉文示例配对）：
  ```json
  [["声", "", ""], ["声", "", "モ"], ["断", "二", "ツ"], ["人", "", "ノ"], ["肠", "一", "ヲ"]]
  [["春", "", ""], ["眠", "", "ネム"], ["不", "", "ズ"], ["觉", "二", "サメ"], ["晓", "一", "アカ"]]
  [["床", "", ""], ["前", "", "マエ"], ["明", "", "アカ"], ["月", "二", "ツキ"], ["光", "一", "ヒカリ"]]
  ```


## 使用方法

### 1. 模型训练
通过`train.py`脚本启动训练，训练过程会自动加载数据、构建标签映射、保存模型权重至`models/`目录。

#### 训练命令
```bash
# 基础训练（使用默认配置）
python train.py

# 可选：指定GPU训练（如使用第0块GPU）
CUDA_VISIBLE_DEVICES=0 python train.py
```

#### 训练配置调整
训练参数（如批次大小、epoch数、模型维度）可在`config.py`中修改，核心配置项：
```python
# 路径配置
TRAIN_HAN_PATH = "data/raw/train/hanwen.txt"    # 训练集汉文路径
TRAIN_KUN_PATH = "data/raw/train/kundoku.jsonl" # 训练集训读路径
VALID_HAN_PATH = "data/raw/valid/hanwen.txt"    # 验证集汉文路径
VALID_KUN_PATH = "data/raw/valid/kundoku.jsonl" # 验证集训读路径
MODEL_SAVE_PATH = "models/kundoku_model.pt"     # 模型保存路径

# 超参数配置
BATCH_SIZE = 32    # 批次大小
EPOCHS = 20        # 训练轮数
LEARNING_RATE = 1e-3 # 学习率
EMBED_DIM = 128    # 汉字嵌入维度
HIDDEN_DIM = 256   # LSTM隐藏层维度
```

#### 训练输出
- 训练过程中会打印每轮的「训练损失」和「验证损失」：
  ```
  Epoch 1/20 | Train Loss: 1.2345 | Valid Loss: 1.1234
  Epoch 2/20 | Train Loss: 0.8765 | Valid Loss: 0.7654
  ...
  ```
- 训练结束后，模型权重会保存至`models/kundoku_model.pt`，标签映射（`order_tag2id.json`、`kana_tag2id.json`）会保存至`data/processed/`。


### 2. 模型推理
通过`predict_demo.py`脚本调用训练好的模型，实现汉文到训读格式的实时转换；也可集成`src/predictor.py`中的`KundokuPredictor`类到其他系统。

#### 推理命令
```bash
# 运行推理示例
python predict_demo.py
```

#### 推理示例代码（predict_demo.py）
```python
from src.predictor import KundokuPredictor
import config

# 初始化预测器（自动加载模型和标签映射）
predictor = KundokuPredictor(
    model_path=config.MODEL_SAVE_PATH,
    tag_mapping_dir=config.TAG_MAPPING_DIR
)

# 输入汉文句子（支持单句或多句列表）
test_han_sentences = [
    "声声断人肠",
    "春眠不觉晓"
]

# 执行预测
for han in test_han_sentences:
    kundoku_result = predictor.predict(han)
    print(f"汉文：{han}")
    print(f"训读格式：{kundoku_result}\n")
```

#### 推理输出
```
汉文：声声断人肠
训读格式：[["声", "", ""], ["声", "", "モ"], ["断", "二", "ツ"], ["人", "", "ノ"], ["肠", "一", "ヲ"]]

汉文：春眠不觉晓
训读格式：[["春", "", ""], ["眠", "", "ネム"], ["不", "", "ズ"], ["觉", "二", "サメ"], ["晓", "一", "アカ"]]
```


## 注意事项
1. 数据质量：确保`hanwen.txt`与`kundoku.jsonl`的句子行数一致，且每个句子的汉字数与三元组列表长度一致，否则会被过滤。
2. 模型升级：若需提升精度，可在`src/model.py`中替换为预训练模型（如`bert-base-chinese`），需同步修改`data_utils.py`中的输入处理逻辑。
3. 推理速度：批量推理时可修改`predictor.py`，支持多句子批量输入，减少GPU/CPU调用开销。
4. 模型保存：训练中断后，可通过加载`models/kundoku_model.pt`继续训练（需在`train.py`中添加断点续训逻辑）。


## 联系方式
若遇到问题或需要功能扩展，可通过以下方式反馈：
- 项目维护者：陈昱充

- 邮箱：chenyuchong2005@163.com
