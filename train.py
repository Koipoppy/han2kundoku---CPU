from src.data_utils import (
    load_data,
    build_tag_mappings,
    save_tag_mappings,
    KundokuDataset,
    collate_fn
)
from src.model import KundokuModel
from src.trainer import train_model
from torch.utils.data import DataLoader
import config  # 正确引用config
import os
import torch

def main():
    print("=" * 60)
    print("===== 汉文训读模型训练入口 =====")
    print("=" * 60)
    
    # 1. 设备配置（与config一致）
    device = torch.device(config.DEVICE)
    print(f"\n[1/6] 设备配置：{device}（CPU训练模式）")
    
    # 2. 加载训练集和验证集（使用config路径）
    print(f"\n[2/6] 加载数据...")
    try:
        train_data = load_data(config.TRAIN_HAN_PATH, config.TRAIN_KUN_PATH)
        valid_data = load_data(config.VALID_HAN_PATH, config.VALID_KUN_PATH)
    except FileNotFoundError as e:
        print(f"数据加载失败：{str(e)}")
        print(f"请确保数据文件存在于：{config.RAW_DATA_DIR}")
        return
    
    print(f"   训练集样本数：{len(train_data)}")
    print(f"   验证集样本数：{len(valid_data)}")
    if len(train_data) == 0 or len(valid_data) == 0:
        print("警告：训练集或验证集为空，无法训练")
        return
    
    # 3. 构建并保存标签映射
    print(f"\n[3/6] 构建标签映射...")
    order_tag2id, kana_tag2id = build_tag_mappings(train_data + valid_data)
    save_tag_mappings(order_tag2id, kana_tag2id)  # 使用config路径
    print(f"   序号标签数量：{len(order_tag2id)}（含<PAD>和<UNK>）")
    print(f"   假名标签数量：{len(kana_tag2id)}（含<PAD>和<UNK>）")
    print(f"   标签映射保存路径：{config.TAG_MAPPING_DIR}")
    
    # 4. 构建数据集和DataLoader（CPU适配num_workers=0）
    print(f"\n[4/6] 构建数据集...")
    train_dataset = KundokuDataset(train_data, order_tag2id, kana_tag2id)
    valid_dataset = KundokuDataset(valid_data, order_tag2id, kana_tag2id)
    
    # 训练集DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # CPU禁用多线程，避免Windows报错
        pin_memory=False
    )
    
    # 验证集DataLoader
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"   训练集批次数量：{len(train_loader)}（批次大小：{config.BATCH_SIZE}）")
    print(f"   验证集批次数量：{len(valid_loader)}（批次大小：{config.BATCH_SIZE}）")
    
    # 5. 初始化模型（参数从config读取）
    print(f"\n[5/6] 初始化模型...")
    model = KundokuModel(
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        order_num_tags=len(order_tag2id),
        kana_num_tags=len(kana_tag2id)
    )
    model.to(device)
    print(f"   模型结构：{model}")
    
    # 6. 开始训练
    print(f"\n[6/6] 开始训练（共{config.EPOCHS}轮）...")
    print("=" * 60)
    try:
        train_model(model, train_loader, valid_loader, order_tag2id, kana_tag2id)
    except Exception as e:
        print(f"\n训练异常终止：{str(e)}")
        return

if __name__ == "__main__":
    main()