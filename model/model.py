import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from pytorch_lightning import LightningModule
import torchmetrics
from typing import Tuple
import timm  # 导入 timm 库
import torch.nn.functional as F


class CatFaceModule(LightningModule):
    def __init__(self, num_classes: int, lr: float, dropout_rate: float = 0.2):  # 添加 dropout_rate 参数
        super(CatFaceModule, self).__init__()

        self.save_hyperparameters()

        # 使用 ConvNeXt-Tiny 作为 backbone
        self.net = timm.create_model('convnext_tiny', pretrained=False, num_classes=0,
                                      global_pool='avg')  # 取消默认分类头，使用自定义分类头

        # 自动推断输入维度 (改进版本)
        if hasattr(self.net, 'classifier') and hasattr(self.net.classifier, 'in_features'):
            in_features = self.net.classifier.in_features # For models with a classifier attribute
        elif hasattr(self.net, 'head') and hasattr(self.net.head, 'fc'):
            in_features = self.net.head.fc.in_features # ConvNeXt style
        elif hasattr(self.net, 'fc') and hasattr(self.net, 'global_pool'):
            # Adaptive pooling but no head, need to determine the number of features manually
            sample_tensor = torch.randn(1, 3, 224, 224)
            features = self.net(sample_tensor).shape[1]  # Extract number of features from the tensor.
            in_features = features
        else:
             in_features = 768  # Provide a default to avoid crash

        # 自定义分类头
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),  # 添加 BatchNorm
            nn.Dropout(dropout_rate),  # 添加 Dropout
            nn.Linear(in_features, num_classes)
        )

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x)
        return self.classifier(features)

    def training_step(self, batch: Tuple[torch.Tensor, torch.LongTensor], batch_idx: int) -> torch.Tensor:
        loss, acc = self.do_step(batch)

        self.log('train/loss', loss, on_step=True, on_epoch=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, acc = self.do_step(batch)

        self.log('val/loss', loss, on_step=False, on_epoch=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True)

    def do_step(self, batch: Tuple[torch.Tensor, torch.LongTensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # shape: x (B, C, H, W), y (B), w (B)
        x, y = batch

        # shape: out (B, num_classes)
        out = self(x)

        loss = self.loss_func(out, y)

        with torch.no_grad():
            # 每个类别分别计算准确率，以平衡地综合考虑每只猫的准确率
            accuracy_per_class = torchmetrics.functional.accuracy(out, y, task="multiclass",
                                                                   num_classes=self.hparams['num_classes'], average=None)
            # 去掉batch中没有出现的类别，这些位置为nan
            nan_mask = accuracy_per_class.isnan()
            accuracy_per_class = accuracy_per_class.masked_fill(nan_mask, 0)
            # 剩下的位置取均值
            acc = accuracy_per_class.sum() / (~nan_mask).sum()

        return loss, acc

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.hparams['lr'])