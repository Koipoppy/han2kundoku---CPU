import json
import os
import torch
from src.model import KundokuModel
import config  # 正确引用config

class KundokuPredictor:
    def __init__(self):
        # 1. 加载标签映射（使用config定义的路径）
        self.order_tag2id = self._load_tag_mapping(config.ORDER_TAG_PATH)
        self.kana_tag2id = self._load_tag_mapping(config.KANA_TAG_PATH)
        
        # 2. 构建反向映射（ID→标签）
        self.id2order_tag = {v: k for k, v in self.order_tag2id.items()}
        self.id2kana_tag = {v: k for k, v in self.kana_tag2id.items()}
        
        # 3. 加载训练好的模型
        self.model = self._load_model()
        
        # 4. 设备配置（与训练一致：CPU）
        self.device = torch.device(config.DEVICE)
        self.model.to(self.device)
        self.model.eval()  # 切换到推理模式

    def _load_tag_mapping(self, mapping_path: str) -> dict:
        """加载标签映射文件（从config指定路径）"""
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"标签映射文件不存在：{mapping_path}\n请先运行train.py训练模型")
        with open(mapping_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_model(self) -> KundokuModel:
        """加载模型权重（CPU专用）"""
        if not os.path.exists(config.MODEL_SAVE_PATH):
            raise FileNotFoundError(f"模型文件不存在：{config.MODEL_SAVE_PATH}\n请先运行train.py训练模型")
        
        # 初始化模型（参数与训练时一致）
        model = KundokuModel(
            vocab_size=config.VOCAB_SIZE,
            embed_dim=config.EMBED_DIM,
            hidden_dim=config.HIDDEN_DIM,
            order_num_tags=len(self.order_tag2id),
            kana_num_tags=len(self.kana_tag2id)
        )
        
        # 加载权重（强制CPU，避免设备不匹配）
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def predict(self, han_sentence: str) -> list:
        """核心推理：输入汉文句子 → 输出训读三元组列表"""
        # 校验输入有效性
        if not isinstance(han_sentence, str) or len(han_sentence) == 0:
            raise ValueError("输入必须为非空汉文句子（字符串类型）")
        
        # 1. 预处理输入：汉字转Unicode ID
        input_ids = torch.tensor([ord(c) for c in han_sentence], dtype=torch.long)
        
        # 2. 增加batch维度（模型要求输入：[batch_size, seq_len]）
        input_ids = input_ids.unsqueeze(0).to(self.device)  # 形状：[1, 句子长度]
        
        # 3. 模型推理（禁用梯度计算，加速+省内存）
        with torch.no_grad():
            outputs = self.model(input_ids)
            # 取概率最大的标签ID（按最后一维）
            order_pred_ids = torch.argmax(outputs["order_logits"], dim=-1).squeeze(0)
            kana_pred_ids = torch.argmax(outputs["kana_logits"], dim=-1).squeeze(0)
        
        # 4. ID转标签（生成最终结果）
        kundoku_result = []
        for char, order_id, kana_id in zip(han_sentence, order_pred_ids, kana_pred_ids):
            order_tag = self.id2order_tag[int(order_id)]
            kana_tag = self.id2kana_tag[int(kana_id)]
            kundoku_result.append([char, order_tag, kana_tag])
        
        return kundoku_result

# 测试代码（直接运行predictor.py可验证）
if __name__ == "__main__":
    try:
        predictor = KundokuPredictor()
        test_han = "风缓雨柔柔"  # 示例输入（来自hanwen.txt）
        result = predictor.predict(test_han)
        
        # 打印结果
        print("=" * 60)
        print(f"汉文输入：{test_han}")
        print(f"训读输出：{result}")
        print("=" * 60)
        
        # 格式化输出（更易读）
        print("\n格式化结果：")
        for idx, (char, order, kana) in enumerate(result, 1):
            print(f"第{idx}字：汉字={char:2s} | 序号标签={order:2s} | 假名标签={kana:4s}")
    except Exception as e:
        print(f"推理失败：{str(e)}")