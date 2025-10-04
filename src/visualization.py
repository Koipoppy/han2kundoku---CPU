import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import config  # 正确引用config
from typing import Optional

# 设置中文字体（避免乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")  # 浅色网格背景

class TrainVisualizer:
    def __init__(self, save_dir: Optional[str] = None):
        """初始化可视化工具：默认使用config的路径"""
        self.save_dir = save_dir if save_dir is not None else config.VISUALIZATION_DIR
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 初始化训练日志（与trainer的指标对应）
        self.train_log: dict = {
            "epoch": [],          # 训练轮次
            "train_total_loss": [],# 训练总损失
            "val_total_loss": [],  # 验证总损失
            "train_order_loss": [],# 训练序号损失
            "val_order_loss": [],  # 验证序号损失
            "train_kana_loss": [], # 训练假名损失
            "val_kana_loss": [],   # 验证假名损失
            "lr": [],              # 学习率
            "train_order_acc": [], # 训练序号准确率
            "val_order_acc": [],   # 验证序号准确率
            "train_kana_acc": [],  # 训练假名准确率
            "val_kana_acc": []     # 验证假名准确率
        }

    def log_epoch_data(self, epoch: int, metrics: dict) -> None:
        """记录每轮训练指标"""
        if not isinstance(epoch, int) or epoch < 1:
            print(f"⚠️  无效轮次（{epoch}），跳过记录")
            return
        self.train_log["epoch"].append(epoch)
        
        valid_keys = list(self.train_log.keys())[1:]
        for key, value in metrics.items():
            if key in valid_keys and isinstance(value, (int, float)):
                self.train_log[key].append(value)
            elif key in valid_keys:
                print(f"⚠️  指标{key}类型错误（{type(value)}），跳过记录")

    def plot_loss_curves(self) -> None:
        """绘制损失曲线（总损失、序号损失、假名损失）"""
        if len(self.train_log["epoch"]) == 0:
            print("⚠️  无训练数据，无法绘制损失曲线")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("模型损失曲线（训练 vs 验证）", fontsize=16, fontweight="bold")
        
        # 1. 总损失曲线
        axes[0].plot(self.train_log["epoch"], self.train_log["train_total_loss"],
                    label="训练总损失", color="#e74c3c", linewidth=2.5, marker="o", markersize=4)
        axes[0].plot(self.train_log["epoch"], self.train_log["val_total_loss"],
                    label="验证总损失", color="#3498db", linewidth=2.5, marker="s", markersize=4)
        axes[0].set_xlabel("轮次")
        axes[0].set_ylabel("平均损失")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)
        
        # 2. 序号损失曲线
        axes[1].plot(self.train_log["epoch"], self.train_log["train_order_loss"],
                    label="训练序号损失", color="#f39c12", linewidth=2.5, marker="o", markersize=4)
        axes[1].plot(self.train_log["epoch"], self.train_log["val_order_loss"],
                    label="验证序号损失", color="#2ecc71", linewidth=2.5, marker="s", markersize=4)
        axes[1].set_xlabel("轮次")
        axes[1].set_ylabel("序号损失")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)
        
        # 3. 假名损失曲线
        axes[2].plot(self.train_log["epoch"], self.train_log["train_kana_loss"],
                    label="训练假名损失", color="#9b59b6", linewidth=2.5, marker="o", markersize=4)
        axes[2].plot(self.train_log["epoch"], self.train_log["val_kana_loss"],
                    label="验证假名损失", color="#1abc9c", linewidth=2.5, marker="s", markersize=4)
        axes[2].set_xlabel("轮次")
        axes[2].set_ylabel("假名损失")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)
        
        # 保存图片
        save_path = os.path.join(self.save_dir, "loss_curves.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ 损失曲线已保存至：{save_path}")

    def plot_lr_curve(self) -> None:
        """绘制学习率衰减曲线"""
        if len(self.train_log["epoch"]) == 0 or len(self.train_log["lr"]) == 0:
            print("⚠️  无学习率数据，无法绘制曲线")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.train_log["epoch"], self.train_log["lr"],
                color="#e67e22", linewidth=3, marker="o", markersize=5, label="学习率")
        
        ax.set_title("学习率衰减曲线", fontsize=14, fontweight="bold")
        ax.set_xlabel("轮次", fontsize=12)
        ax.set_ylabel("学习率", fontsize=12)
        ax.set_yscale("log")  # 对数坐标，清晰展示衰减
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
        # 保存图片
        save_path = os.path.join(self.save_dir, "lr_curve.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ 学习率曲线已保存至：{save_path}")

    def plot_accuracy_curves(self) -> None:
        """绘制准确率曲线（序号、假名）"""
        if len(self.train_log["epoch"]) == 0:
            print("⚠️  无训练数据，无法绘制准确率曲线")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("模型准确率曲线（训练 vs 验证）", fontsize=16, fontweight="bold")
        
        # 1. 序号准确率曲线
        axes[0].plot(self.train_log["epoch"], self.train_log["train_order_acc"],
                    label="训练序号准确率", color="#e74c3c", linewidth=2.5, marker="o", markersize=4)
        axes[0].plot(self.train_log["epoch"], self.train_log["val_order_acc"],
                    label="验证序号准确率", color="#3498db", linewidth=2.5, marker="s", markersize=4)
        axes[0].set_xlabel("轮次")
        axes[0].set_ylabel("准确率 (%)")
        axes[0].set_ylim(0, 105)  # 固定y轴范围，便于对比
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.3)
        
        # 2. 假名准确率曲线
        axes[1].plot(self.train_log["epoch"], self.train_log["train_kana_acc"],
                    label="训练假名准确率", color="#9b59b6", linewidth=2.5, marker="o", markersize=4)
        axes[1].plot(self.train_log["epoch"], self.train_log["val_kana_acc"],
                    label="验证假名准确率", color="#1abc9c", linewidth=2.5, marker="s", markersize=4)
        axes[1].set_xlabel("轮次")
        axes[1].set_ylabel("准确率 (%)")
        axes[1].set_ylim(0, 105)
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)
        
        # 保存图片
        save_path = os.path.join(self.save_dir, "accuracy_curves.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ 准确率曲线已保存至：{save_path}")

    def plot_confusion_matrix(self, true_labels: list, pred_labels: list, tag2id: dict, task_name: str) -> None:
        """绘制混淆矩阵（解决set_label None错误）"""
        # 校验输入有效性
        if not (isinstance(true_labels, list) and isinstance(pred_labels, list)):
            print(f"⚠️  {task_name}混淆矩阵：标签列表必须为list，跳过绘制")
            return
        if not isinstance(tag2id, dict) or len(tag2id) == 0:
            print(f"⚠️  {task_name}混淆矩阵：标签映射无效，跳过绘制")
            return
        if len(true_labels) == 0 or len(pred_labels) == 0:
            print(f"⚠️  {task_name}混淆矩阵：无有效标签，跳过绘制")
            return
        
        # 过滤PAD标签（=0）
        valid_mask = [t != 0 for t in true_labels]
        filtered_true = [true_labels[i] for i, mask in enumerate(valid_mask) if mask]
        filtered_pred = [pred_labels[i] for i, mask in enumerate(valid_mask) if mask]
        if len(filtered_true) == 0:
            print(f"⚠️  {task_name}混淆矩阵：过滤后无有效标签，跳过绘制")
            return
        
        # 构建标签映射（ID→标签）
        id2tag = {v: k for k, v in tag2id.items()}
        sorted_ids = sorted(tag2id.values())  # 按ID排序，确保行列对齐
        tag_names = [id2tag[tag_id] for tag_id in sorted_ids]
        
        # 计算混淆矩阵（归一化到百分比）
        cm = confusion_matrix(filtered_true, filtered_pred, labels=sorted_ids)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # 绘制混淆矩阵
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = sns.heatmap(
            cm_normalized,
            annot=False,  # 标签过多时禁用标注，避免拥挤
            cmap="YlOrRd",
            fmt=".1f",
            xticklabels=tag_names,
            yticklabels=tag_names,
            ax=ax
        )
        
        ax.set_title(f"混淆矩阵 - {task_name}任务（验证集）",
                    fontsize=14, fontweight="bold", pad=20)
        ax.set_xlabel("预测标签", fontsize=12)
        ax.set_ylabel("真实标签", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        
        # 处理颜色条（避免None错误）
        if heatmap.collections:
            cbar = heatmap.collections[0].colorbar
            if cbar is not None:  # 增加None判断
                cbar.set_label("准确率 (%)", rotation=270, labelpad=20, fontsize=12)
        
        # 保存图片
        save_path = os.path.join(self.save_dir, f"confusion_matrix_{task_name}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ {task_name}混淆矩阵已保存至：{save_path}")

    def generate_all_plots(self, val_true_order: list, val_pred_order: list,
                          val_true_kana: list, val_pred_kana: list,
                          order_tag2id: dict, kana_tag2id: dict) -> None:
        """生成所有可视化图表"""
        print("\n===== 开始生成训练可视化图表 =====")
        # 确保输入为有效列表
        val_true_order = val_true_order if isinstance(val_true_order, list) else []
        val_pred_order = val_pred_order if isinstance(val_pred_order, list) else []
        val_true_kana = val_true_kana if isinstance(val_true_kana, list) else []
        val_pred_kana = val_pred_kana if isinstance(val_pred_kana, list) else []
        order_tag2id = order_tag2id if isinstance(order_tag2id, dict) else {}
        kana_tag2id = kana_tag2id if isinstance(kana_tag2id, dict) else {}
        
        # 生成所有图表
        self.plot_loss_curves()
        self.plot_lr_curve()
        self.plot_accuracy_curves()
        self.plot_confusion_matrix(val_true_order, val_pred_order, order_tag2id, "序号预测")
        self.plot_confusion_matrix(val_true_kana, val_pred_kana, kana_tag2id, "假名预测")
        
        print(f"===== 所有图表已保存至：{self.save_dir} =====")