import json
import os
import torch
from src.model import KundokuModel  # 导入模型类
import config  # 导入配置

class KundokuPredictor:
    def __init__(self):
        # 1. 加载标签映射（训练时保存的order_tag2id和kana_tag2id）
        self.order_tag2id = self._load_tag_mapping(config.TAG_MAPPING_DIR, "order_tag2id.json")
        self.kana_tag2id = self._load_tag_mapping(config.TAG_MAPPING_DIR, "kana_tag2id.json")
        # 2. 构建反向映射（id→标签，用于将模型输出的id转成序号/假名）
        self.id2order_tag = {v: k for k, v in self.order_tag2id.items()}
        self.id2kana_tag = {v: k for k, v in self.kana_tag2id.items()}
        # 3. 加载训练好的模型
        self.model = self._load_model()
        # 4. 设备配置（与训练设备一致）
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()  # 切换到推理模式（关闭dropout）

    def _load_tag_mapping(self, mapping_dir: str, filename: str) -> dict:
        """加载标签映射文件（order_tag2id.json/kana_tag2id.json）"""
        mapping_path = os.path.join(mapping_dir, filename)
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"标签映射文件不存在：{mapping_path}，请先训练模型")
        with open(mapping_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_model(self) -> KundokuModel:
        """加载训练好的模型权重"""
        if not os.path.exists(config.MODEL_SAVE_PATH):
            raise FileNotFoundError(f"模型文件不存在：{config.MODEL_SAVE_PATH}，请先训练模型")
        # 初始化模型（参数与训练时一致）
        model = KundokuModel(
            vocab_size=config.VOCAB_SIZE,
            embed_dim=config.EMBED_DIM,
            hidden_dim=config.HIDDEN_DIM,
            order_num_tags=len(self.order_tag2id),
            kana_num_tags=len(self.kana_tag2id)
        )
        # 加载模型权重（处理检查点格式）
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location="cpu")
        if "model_state_dict" in checkpoint:
            # 如果是完整检查点，提取模型状态字典
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # 如果是纯状态字典
            model.load_state_dict(checkpoint)
        return model

    def predict(self, han_sentence: str) -> list:
        """核心推理函数：输入汉文句子，输出训读格式（三元组列表）"""
        # 1. 预处理输入：汉文句子转Unicode ID（与训练时一致）
        input_ids = torch.tensor([ord(c) for c in han_sentence], dtype=torch.long)
        # 2. 增加batch维度（模型要求输入格式：[batch_size, seq_len]）
        input_ids = input_ids.unsqueeze(0).to(self.device)  # 形状：[1, 句子长度]

        # 3. 模型推理（禁用梯度计算，加速+省内存）
        with torch.no_grad():
            outputs = self.model(input_ids)
            # 取预测结果（argmax：选择概率最大的标签ID）
            order_pred_ids = torch.argmax(outputs["order_logits"], dim=-1).squeeze(0)  # 序号预测ID
            kana_pred_ids = torch.argmax(outputs["kana_logits"], dim=-1).squeeze(0)    # 假名预测ID

        # 4. ID转标签（将模型输出的ID映射为实际的序号/假名）
        kundoku_result = []
        for char, order_id, kana_id in zip(han_sentence, order_pred_ids, kana_pred_ids):
            order_tag = self.id2order_tag[int(order_id)]
            kana_tag = self.id2kana_tag[int(kana_id)]
            kundoku_result.append([char, order_tag, kana_tag])

        return kundoku_result


# 测试代码
if __name__ == "__main__":
    # 初始化预测器
    predictor = KundokuPredictor()
    
    # 测试新的汉文句子（可替换成你想测试的句子）
    test_han = "声声人断肠"  
    result = predictor.predict(test_han)
    
    # 打印结果
    print(f"汉文输入：{test_han}")
    print(f"训读输出：{result}")