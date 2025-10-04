# config.py - 完整定义所有依赖属性，确保无缺失
import os

# ---------------------- 项目根目录配置 ----------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))

# ---------------------- 数据路径配置 ----------------------
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")  # 原始数据目录
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")  # 处理后数据目录
TAG_MAPPING_DIR = PROCESSED_DATA_DIR  # 标签映射保存目录（复用processed）

# 训练/验证数据路径（严格对应项目结构）
TRAIN_HAN_PATH = os.path.join(RAW_DATA_DIR, "train", "hanwen.txt")
TRAIN_KUN_PATH = os.path.join(RAW_DATA_DIR, "train", "kundoku.jsonl")
VALID_HAN_PATH = os.path.join(RAW_DATA_DIR, "valid", "hanwen.txt")
VALID_KUN_PATH = os.path.join(RAW_DATA_DIR, "valid", "kundoku.jsonl")

# 标签映射文件路径（供data_utils和predictor使用）
ORDER_TAG_PATH = os.path.join(TAG_MAPPING_DIR, "order_tag2id.json")
KANA_TAG_PATH = os.path.join(TAG_MAPPING_DIR, "kana_tag2id.json")

# ---------------------- 输出路径配置 ----------------------
MODEL_DIR = os.path.join(ROOT_DIR, "models")  # 模型保存目录
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "kundoku_model.pt")  # 模型文件路径
VISUALIZATION_DIR = os.path.join(ROOT_DIR, "visualization")  # 可视化结果目录

# ---------------------- 模型参数配置 ----------------------
VOCAB_SIZE = 50000  # 词汇表大小（支持Unicode汉字范围）
EMBED_DIM = 128  # 汉字嵌入维度
HIDDEN_DIM = 256  # LSTM隐藏层维度
MAX_SEQ_LEN = 100  # 最大序列长度
DROPOUT_RATE = 0.3  # Dropout概率（模型中已禁用，保留配置）

# ---------------------- 训练参数配置 ----------------------
BATCH_SIZE = 16  # CPU适配：减小批次避免内存不足
EPOCHS = 20  # 训练轮次
LEARNING_RATE = 5e-5  # 初始学习率
DEVICE = "cpu"  # 强制CPU训练
STEP_SIZE = 5  # 学习率衰减步长（每5轮衰减）
GAMMA = 0.8  # 学习率衰减系数（每次衰减为原80%）

# ---------------------- 自动创建目录 ----------------------
required_dirs = [
    RAW_DATA_DIR,
    os.path.join(RAW_DATA_DIR, "train"),
    os.path.join(RAW_DATA_DIR, "valid"),
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    VISUALIZATION_DIR
]
for dir_path in required_dirs:
    os.makedirs(dir_path, exist_ok=True)