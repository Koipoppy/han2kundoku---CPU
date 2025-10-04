import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import config  # 正确引用config
from src.visualization import TrainVisualizer

def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """计算准确率（忽略PAD标签=0）"""
    valid_mask = labels != 0
    if valid_mask.sum() == 0:
        return 100.0  # 无有效标签时返回100%
    
    pred_labels = torch.argmax(logits, dim=1)
    correct_count = (pred_labels[valid_mask] == labels[valid_mask]).sum().item()
    accuracy = (correct_count / valid_mask.sum().item()) * 100
    return accuracy

def train_model(model, train_loader, valid_loader, order_tag2id, kana_tag2id):
    """完整训练逻辑（CPU适配，无CUDA依赖）"""
    # 1. 初始化可视化工具（使用config的路径）
    visualizer = TrainVisualizer(save_dir=config.VISUALIZATION_DIR)
    
    # 2. 设备配置（强制CPU，与config一致）
    device = torch.device(config.DEVICE)
    model = model.to(device)
    print(f"===== 训练设备：{device} =====")
    
    # 3. 训练组件初始化
    # 损失函数：忽略PAD标签=0（与数据处理逻辑一致）
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # 优化器：AdamW（带L2正则化）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=1e-4
    )
    # 学习率调度器（按config配置）
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.STEP_SIZE,
        gamma=config.GAMMA
    )
    
    # 4. 训练状态初始化
    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    
    # 5. 核心训练循环
    for epoch in range(1, config.EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n===== 训练轮次 {epoch}/{config.EPOCHS} =====")
        print(f"当前学习率：{current_lr:.6f}")
        
        # ---------------------- 训练阶段 ----------------------
        model.train()
        train_metrics = {
            "total_loss": 0.0, "order_loss": 0.0, "kana_loss": 0.0,
            "order_acc": 0.0, "kana_acc": 0.0, "sample_count": 0
        }
        train_bar = tqdm(train_loader, desc=f"训练批次 [Epoch {epoch}]")
        
        for batch in train_bar:
            # 数据移至CPU
            input_ids = batch["input_ids"].to(device)
            order_labels = batch["order_labels"].to(device)
            kana_labels = batch["kana_labels"].to(device)
            batch_size = input_ids.size(0)
            train_metrics["sample_count"] += batch_size
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播（无混合精度，CPU无需）
            outputs = model(input_ids)
            order_logits = outputs["order_logits"]
            kana_logits = outputs["kana_logits"]
            
            # 展平数据（适配CrossEntropyLoss输入格式）
            order_logits_flat = order_logits.reshape(-1, order_logits.shape[-1])
            kana_logits_flat = kana_logits.reshape(-1, kana_logits.shape[-1])
            order_labels_flat = order_labels.reshape(-1)
            kana_labels_flat = kana_labels.reshape(-1)
            
            # 计算损失与准确率
            order_loss = criterion(order_logits_flat, order_labels_flat)
            kana_loss = criterion(kana_logits_flat, kana_labels_flat)
            total_loss = (order_loss + kana_loss) / 2  # 双任务损失平均
            order_acc = calculate_accuracy(order_logits_flat, order_labels_flat)
            kana_acc = calculate_accuracy(kana_logits_flat, kana_labels_flat)
            
            # 反向传播与参数更新
            total_loss.backward()
            optimizer.step()
            
            # 累计指标
            train_metrics["total_loss"] += total_loss.item() * batch_size
            train_metrics["order_loss"] += order_loss.item() * batch_size
            train_metrics["kana_loss"] += kana_loss.item() * batch_size
            train_metrics["order_acc"] += order_acc * batch_size
            train_metrics["kana_acc"] += kana_acc * batch_size
            
            # 更新进度条
            train_bar.set_postfix({
                "总损失": f"{total_loss.item():.4f}",
                "序号损失": f"{order_loss.item():.4f}",
                "假名损失": f"{kana_loss.item():.4f}",
                "序号准确率": f"{order_acc:.2f}%",
                "假名准确率": f"{kana_acc:.2f}%"
            })
        
        # 计算训练集平均指标
        train_avg_total_loss = train_metrics["total_loss"] / train_metrics["sample_count"]
        train_avg_order_loss = train_metrics["order_loss"] / train_metrics["sample_count"]
        train_avg_kana_loss = train_metrics["kana_loss"] / train_metrics["sample_count"]
        train_avg_order_acc = train_metrics["order_acc"] / train_metrics["sample_count"]
        train_avg_kana_acc = train_metrics["kana_acc"] / train_metrics["sample_count"]
        
        print(f"\n【训练集结果】")
        print(f"总损失：{train_avg_total_loss:.4f} | 序号损失：{train_avg_order_loss:.4f} | 假名损失：{train_avg_kana_loss:.4f}")
        print(f"序号准确率：{train_avg_order_acc:.2f}% | 假名准确率：{train_avg_kana_acc:.2f}%")
        
        # ---------------------- 验证阶段 ----------------------
        model.eval()
        val_metrics = {
            "total_loss": 0.0, "order_loss": 0.0, "kana_loss": 0.0,
            "order_acc": 0.0, "kana_acc": 0.0, "sample_count": 0
        }
        # 初始化标签列表（用于混淆矩阵）
        val_true_order = []
        val_pred_order = []
        val_true_kana = []
        val_pred_kana = []
        
        with torch.no_grad():  # 禁用梯度计算
            val_bar = tqdm(valid_loader, desc=f"验证批次 [Epoch {epoch}]")
            
            for batch in val_bar:
                # 数据移至CPU
                input_ids = batch["input_ids"].to(device)
                order_labels = batch["order_labels"].to(device)
                kana_labels = batch["kana_labels"].to(device)
                batch_size = input_ids.size(0)
                val_metrics["sample_count"] += batch_size
                
                # 模型预测
                outputs = model(input_ids)
                order_logits = outputs["order_logits"]
                kana_logits = outputs["kana_logits"]
                
                # 展平数据
                order_logits_flat = order_logits.reshape(-1, order_logits.shape[-1])
                kana_logits_flat = kana_logits.reshape(-1, kana_logits.shape[-1])
                order_labels_flat = order_labels.reshape(-1)
                kana_labels_flat = kana_labels.reshape(-1)
                
                # 计算损失与准确率
                order_loss = criterion(order_logits_flat, order_labels_flat)
                kana_loss = criterion(kana_logits_flat, kana_labels_flat)
                total_loss = (order_loss + kana_loss) / 2
                order_acc = calculate_accuracy(order_logits_flat, order_labels_flat)
                kana_acc = calculate_accuracy(kana_logits_flat, kana_labels_flat)
                
                # 累计指标
                val_metrics["total_loss"] += total_loss.item() * batch_size
                val_metrics["order_loss"] += order_loss.item() * batch_size
                val_metrics["kana_loss"] += kana_loss.item() * batch_size
                val_metrics["order_acc"] += order_acc * batch_size
                val_metrics["kana_acc"] += kana_acc * batch_size
                
                # 记录标签（过滤PAD）
                valid_mask = order_labels_flat != 0
                val_true_order.extend(order_labels_flat[valid_mask].cpu().numpy().tolist())
                val_pred_order.extend(torch.argmax(order_logits_flat[valid_mask], dim=1).cpu().numpy().tolist())
                val_true_kana.extend(kana_labels_flat[valid_mask].cpu().numpy().tolist())
                val_pred_kana.extend(torch.argmax(kana_logits_flat[valid_mask], dim=1).cpu().numpy().tolist())
                
                # 更新进度条
                val_bar.set_postfix({
                    "总损失": f"{total_loss.item():.4f}",
                    "序号损失": f"{order_loss.item():.4f}",
                    "假名损失": f"{kana_loss.item():.4f}"
                })
        
        # 计算验证集平均指标
        val_avg_total_loss = val_metrics["total_loss"] / val_metrics["sample_count"]
        val_avg_order_loss = val_metrics["order_loss"] / val_metrics["sample_count"]
        val_avg_kana_loss = val_metrics["kana_loss"] / val_metrics["sample_count"]
        val_avg_order_acc = val_metrics["order_acc"] / val_metrics["sample_count"]
        val_avg_kana_acc = val_metrics["kana_acc"] / val_metrics["sample_count"]
        
        print(f"\n【验证集结果】")
        print(f"总损失：{val_avg_total_loss:.4f} | 序号损失：{val_avg_order_loss:.4f} | 假名损失：{val_avg_kana_loss:.4f}")
        print(f"序号准确率：{val_avg_order_acc:.2f}% | 假名准确率：{val_avg_kana_acc:.2f}%")
        
        # ---------------------- 模型保存与可视化 ----------------------
        # 保存最优模型（按验证损失）
        if val_avg_total_loss < best_val_loss:
            best_val_loss = val_avg_total_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "order_tag2id": order_tag2id,
                "kana_tag2id": kana_tag2id
            }, config.MODEL_SAVE_PATH)
            print(f"✅ 保存最优模型至：{config.MODEL_SAVE_PATH}（验证损失：{best_val_loss:.4f}）")
        
        # 记录指标用于可视化
        visualizer.log_epoch_data(
            epoch=epoch,
            metrics={
                "train_total_loss": train_avg_total_loss,
                "val_total_loss": val_avg_total_loss,
                "train_order_loss": train_avg_order_loss,
                "val_order_loss": val_avg_order_loss,
                "train_kana_loss": train_avg_kana_loss,
                "val_kana_loss": val_avg_kana_loss,
                "lr": current_lr,
                "train_order_acc": train_avg_order_acc,
                "val_order_acc": val_avg_order_acc,
                "train_kana_acc": train_avg_kana_acc,
                "val_kana_acc": val_avg_kana_acc
            }
        )
        
        # 更新学习率
        lr_scheduler.step()
    
    # ---------------------- 训练结束：生成可视化图表 ----------------------
    visualizer.generate_all_plots(
        val_true_order=val_true_order,
        val_pred_order=val_pred_order,
        val_true_kana=val_true_kana,
        val_pred_kana=val_pred_kana,
        order_tag2id=order_tag2id,
        kana_tag2id=kana_tag2id
    )
    
    # 训练总结
    print(f"\n===== 训练完成！=====")
    print(f"训练轮次：{config.EPOCHS}")
    print(f"最优验证损失：{best_val_loss:.4f}")
    print(f"模型保存路径：{config.MODEL_SAVE_PATH}")
    print(f"可视化图表路径：{visualizer.save_dir}")