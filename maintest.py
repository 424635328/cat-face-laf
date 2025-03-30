import argparse
import os
import json
import torch
from model.model import CatFaceModule
from model.data import CatPhotoDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def parse_args():
    """解析命令行参数并返回参数对象"""
    parser = argparse.ArgumentParser(description="Cat Recognize Model Trainer")
    parser.add_argument("--data", default="data/crop_photos", type=str, help="Photo data directory")
    parser.add_argument("--size", default=128, type=int, help="Image size for training")
    parser.add_argument("--filter", default=2, type=int, help="Minimum number of photos required per cat")
    parser.add_argument("--balance", default=30, type=int, help="Number of samples per cat per epoch for balancing")
    parser.add_argument("--lr", default=3e-4, type=float, help="Learning rate for the optimizer")
    parser.add_argument("--batch", default=32, type=int, help="Batch size for training")
    parser.add_argument("--epoch", default=10, type=int, help="Total number of epochs to train")
    parser.add_argument("--name", default='cat', type=str, help="Name of the model for saving and exporting")
    return parser.parse_args()

def validate_args(args):
    """验证参数的有效性"""
    if not os.path.exists(args.data):
        raise ValueError(f"Data directory {args.data} does not exist.")
    if args.size <= 0:
        raise ValueError("Image size must be a positive integer.")
    if args.batch <= 0:
        raise ValueError("Batch size must be a positive integer.")
    if args.epoch <= 0:
        raise ValueError("Number of epochs must be a positive integer.")

def setup_logging(model_name):
    """设置训练日志和模型检查点回调"""
    logger = TensorBoardLogger('./', version=model_name, default_hp_metric=False)
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',        # 保存检查点的目录
        filename=model_name,           # 检查点文件名
        monitor='val/acc',             # 监控验证集准确率
        mode='max'                     # 选择最高的准确率作为最佳模型
    )
    return logger, checkpoint_callback

def export_model(model, model_name, size, cat_ids):
    """导出模型为 ONNX 格式并保存类别 ID 映射"""
    os.makedirs('export/', exist_ok=True)  # 创建 export 目录（如果不存在）
    model.to_onnx(f'export/{model_name}.onnx', torch.randn(1, 3, size, size), export_params=True)
    
    # 保存类别 ID 映射到 JSON 文件
    with open(f'export/{model_name}.json', 'w') as fp:
        json.dump(cat_ids, fp)

def main():
    """主函数，控制训练流程"""
    args = parse_args()  # 解析命令行参数
    validate_args(args)  # 验证参数的有效性

    # 加载数据模块
    data_module = CatPhotoDataModule(args.data, args.size, args.filter, args.balance, args.batch)
    # 创建模型实例
    model = CatFaceModule(len(data_module.cat_ids), args.lr)

    # 检查可用设备
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    print(f'Training device: {device}')  # 输出所用设备

    # 设置日志记录和检查点
    logger, checkpoint_callback = setup_logging(args.name)
    
    # 创建训练器实例
    trainer = Trainer(
        accelerator=device,             # 使用的设备
        devices=1 if device == 'gpu' else None,  # 仅使用单个 GPU
        logger=logger,                  # 日志记录器
        callbacks=[checkpoint_callback], # 检查点回调
        max_epochs=args.epoch           # 最大训练周期数
    )
    
    # 开始训练
    print("Starting training...")
    trainer.fit(model, datamodule=data_module)

    # 从最佳检查点加载模型
    best_model = CatFaceModule.load_from_checkpoint(f'checkpoints/{args.name}.ckpt')
    # 导出模型
    print('Exporting model...')
    export_model(best_model, args.name, args.size, data_module.cat_ids)

    print('Training and export completed.')  # 训练和导出完成

if __name__ == '__main__':
    main()  # 运行主程序
