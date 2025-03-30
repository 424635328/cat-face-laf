#!/bin/bash

# 拉取 YOLOv5 代码
git submodule update --init

# 下载 YOLOv5 需要的模型文件
wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-m-converted.pt -O ./yolov9/yolov9m.pt

# 进入 YOLOv5 项目，导出 ONNX 模型
cd yolov9
python export.py --weights yolov9m.pt --include onnx