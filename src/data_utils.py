import json
import os
from typing import List, Dict, Tuple, Union
import torch
from torch import Tensor
from torch.utils.data import Dataset
import config  # 正确引用config

def load_data(hanwen_path: str, kundoku_path: str) -> List[Dict]:
    """加载汉文与训读配对数据，过滤无效样本"""
    data = []
    try:
        with open(hanwen_path, 'r', encoding='utf-8') as hf, \
             open(kundoku_path, 'r', encoding='utf-8') as kf:
            for line_idx, (han_line, kun_line) in enumerate(zip(hf, kf), 1):
                han = han_line.strip()
                if not han:
                    print(f"警告：第{line_idx}行汉文为空，跳过")
                    continue
                # 解析训读JSON
                try:
                    kun = json.loads(kun_line.strip())
                except json.JSONDecodeError:
                    print(f"警告：第{line_idx}行训读格式错误，跳过")
                    continue
                # 校验长度匹配
                if len(han) != len(kun):
                    print(f"警告：第{line_idx}行汉文与训读长度不匹配（汉文{len(han)}字，训读{len(kun)}项），跳过")
                    continue
                data.append({"han": han, "kun": kun})
    except FileNotFoundError as e:
        raise FileNotFoundError(f"数据文件不存在：{e.filename}") from e
    return data

def build_tag_mappings(data: List[Dict]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """从数据集中提取标签，构建tag2id映射（含<PAD>和<UNK>）"""
    order_tags = set()  # 序号标签（如"", "一", "二"）
    kana_tags = set()   # 假名标签（如"", "モ", "ツ"）
    
    for sample in data:
        for item in sample["kun"]:
            order_tags.add(item[1])
            kana_tags.add(item[2])
    
    # 构建映射：0=PAD，1=UNK
    order_tag2id = {"<PAD>": 0, "<UNK>": 1}
    order_tag2id.update({tag: i + 2 for i, tag in enumerate(sorted(order_tags))})
    
    kana_tag2id = {"<PAD>": 0, "<UNK>": 1}
    kana_tag2id.update({tag: i + 2 for i, tag in enumerate(sorted(kana_tags))})
    
    print(f"标签映射构建完成：序号标签{len(order_tag2id)}个，假名标签{len(kana_tag2id)}个")
    return order_tag2id, kana_tag2id

def save_tag_mappings(order_tag2id: Dict[str, int], kana_tag2id: Dict[str, int]) -> None:
    """保存标签映射到config定义的路径"""
    # 使用config的TAG_MAPPING_DIR，避免硬编码
    os.makedirs(config.TAG_MAPPING_DIR, exist_ok=True)
    
    # 保存序号映射
    with open(config.ORDER_TAG_PATH, "w", encoding="utf-8") as f:
        json.dump(order_tag2id, f, ensure_ascii=False, indent=2)
    
    # 保存假名映射
    with open(config.KANA_TAG_PATH, "w", encoding="utf-8") as f:
        json.dump(kana_tag2id, f, ensure_ascii=False, indent=2)
    
    print(f"标签映射已保存至：{config.TAG_MAPPING_DIR}")

class KundokuDataset(Dataset):
    """汉文训读数据集：将文本转换为模型可输入的Tensor"""
    def __init__(self, data: List[Dict], order_tag2id: Dict[str, int], kana_tag2id: Dict[str, int]):
        self.data = data
        self.order_tag2id = order_tag2id
        self.kana_tag2id = kana_tag2id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, int]]:
        sample = self.data[idx]
        han = sample["han"]
        kun = sample["kun"]
        
        # 汉字转Unicode ID（输入特征）
        input_ids = torch.tensor([ord(c) for c in han], dtype=torch.long)
        
        # 标签转ID（未知标签用<UNK>=1）
        order_labels = torch.tensor(
            [self.order_tag2id.get(item[1], self.order_tag2id["<UNK>"]) for item in kun],
            dtype=torch.long
        )
        kana_labels = torch.tensor(
            [self.kana_tag2id.get(item[2], self.kana_tag2id["<UNK>"]) for item in kun],
            dtype=torch.long
        )
        
        return {
            "input_ids": input_ids,
            "order_labels": order_labels,
            "kana_labels": kana_labels,
            "length": len(han)  # 句子实际长度（用于Padding）
        }

def collate_fn(batch: List[Dict[str, Union[Tensor, int]]]) -> Dict[str, Tensor]:
    """批量处理：统一句子长度（短句补PAD=0）"""
    max_len = max(int(item["length"]) for item in batch)
    input_ids_list = []
    order_labels_list = []
    kana_labels_list = []
    
    for item in batch:
        # 验证数据类型
        input_ids = item["input_ids"]
        assert isinstance(input_ids, Tensor), f"input_ids必须为Tensor，实际为{type(input_ids)}"
        
        order_labels = item["order_labels"]
        assert isinstance(order_labels, Tensor), f"order_labels必须为Tensor，实际为{type(order_labels)}"
        
        kana_labels = item["kana_labels"]
        assert isinstance(kana_labels, Tensor), f"kana_labels必须为Tensor，实际为{type(kana_labels)}"
        
        pad_len = max_len - int(item["length"])
        assert pad_len >= 0, f"填充长度不能为负（句子长度：{item['length']}，最大长度：{max_len}）"
        
        # 补PAD（输入和标签均用0，与CrossEntropyLoss的ignore_index对齐）
        input_ids_padded = torch.nn.functional.pad(input_ids, (0, pad_len), value=0)
        order_labels_padded = torch.nn.functional.pad(order_labels, (0, pad_len), value=0)
        kana_labels_padded = torch.nn.functional.pad(kana_labels, (0, pad_len), value=0)
        
        input_ids_list.append(input_ids_padded)
        order_labels_list.append(order_labels_padded)
        kana_labels_list.append(kana_labels_padded)
    
    return {
        "input_ids": torch.stack(input_ids_list),  # [batch_size, max_len]
        "order_labels": torch.stack(order_labels_list),  # [batch_size, max_len]
        "kana_labels": torch.stack(kana_labels_list)   # [batch_size, max_len]
    }