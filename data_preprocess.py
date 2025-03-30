import os
import shutil
import torch
from tqdm import tqdm
from PIL import Image
import random

SRC = 'data/photos'  # 原始照片的根目录
DEST = 'data/crop_photos' # 裁剪后照片的根目录

# 数据增强参数 (根据需要调整)
ROTATION_RANGE = 15  # 旋转角度范围（度）
SCALE_RANGE = 0.1  # 缩放比例范围 (例如 0.1 表示 +/- 10%)
TRANSLATION_RANGE = 0.1  # 平移范围，相对于图像尺寸的百分比
BRIGHTNESS_RANGE = 0.2  # 亮度调整范围
CONTRAST_RANGE = 0.2  # 对比度调整范围
SATURATION_RANGE = 0.2  # 饱和度调整范围
HUE_RANGE = 0.1  # 色调调整范围 (小心使用较大的范围)
NUM_AUGMENTATIONS = 5  # 每张原始图片生成多少张增强图片

def augment_image(image):
    """对 PIL Image 应用随机的数据增强."""
    augmented_images = []
    for _ in range(NUM_AUGMENTATIONS):
        img = image.copy()  # 每次增强都从原始图像的副本开始

        # 1. 几何变换
        rotation = random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
        img = img.rotate(rotation)

        scale = 1 + random.uniform(-SCALE_RANGE, SCALE_RANGE)
        width, height = img.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS) # 使用 Lanczos 算法获得更好的缩放效果

        translation_x = random.uniform(-TRANSLATION_RANGE, TRANSLATION_RANGE) * width
        translation_y = random.uniform(-TRANSLATION_RANGE, TRANSLATION_RANGE) * height
        img = img.transform(img.size, Image.AFFINE, (1, 0, translation_x, 0, 1, translation_y)) # 应用平移变换


        # 2. 颜色变换
        brightness = 1 + random.uniform(-BRIGHTNESS_RANGE, BRIGHTNESS_RANGE)
        contrast = 1 + random.uniform(-CONTRAST_RANGE, CONTRAST_RANGE)
        saturation = 1 + random.uniform(-SATURATION_RANGE, SATURATION_RANGE)
        hue = random.uniform(-HUE_RANGE, HUE_RANGE) # 色调调整，注意控制范围

        img = img.convert('HSV') # 转换到 HSV 色彩空间进行颜色调整
        pixels = img.load()

        for i in range(img.size[0]):
            for j in range(img.size[1]):
                h, s, v = pixels[i, j]

                v = int(v * brightness) # 亮度调整
                v = max(0, min(v, 255))  # 限制取值范围在 0-255

                s = int(s * saturation) # 饱和度调整
                s = max(0, min(s, 255))

                h = int(h + hue * 180)  # 色调调整
                h = h % 180 # 保证色调值在 0-180 范围内

                pixels[i, j] = (h, s, v) # 更新像素值


        img = img.convert('RGB') # 转换回 RGB 色彩空间

        augmented_images.append(img)
    return augmented_images


if __name__ == '__main__':
    print('正在加载 YOLOv9 模型...')
    model = torch.hub.load('yolov9', 'custom', 'yolov9/yolov9m.pt', source='local')
    model.conf = 0.7  # 设置置信度阈值 (例如 0.7)
    model.iou = 0.45 # 设置 IoU 阈值 (例如 0.45)

    num_photos = 0
    num_skipped_photos = 0
    num_augmented_photos = 0

    if os.path.exists(DEST):
        shutil.rmtree(DEST) # 如果目标目录存在，先删除
    os.makedirs(DEST, exist_ok=True)  # 创建目标目录，如果目录已存在则忽略

    print('正在处理照片...')
    for dir_name in tqdm(os.listdir(SRC), leave=False, desc='processing'):
        src_path = os.path.join(SRC, dir_name)
        if not os.path.isdir(src_path):
            continue

        dest_path = os.path.join(DEST, dir_name)
        os.makedirs(dest_path, exist_ok=True)

        for file_name in tqdm(os.listdir(src_path), leave=False, desc=dir_name):
            num_photos += 1

            src_file_path = os.path.join(src_path, file_name)
            base_file_name, ext = os.path.splitext(file_name)  # 分割文件名和扩展名

            try:
                # 使用 YOLOv9 进行目标检测，结果为[{xmin, ymin, xmax, ymax, confidence, class, name}]格式
                results = model(src_file_path).pandas().xyxy[0].to_dict('records')
            except OSError as err:
                # 发现有的图片有问题，会导致 PIL 抛出 OSError: image file is truncated
                num_skipped_photos += 1
                continue

            # 过滤非cat目标
            cat_results = list(filter(lambda target: target['name'] == 'cat', results))

            # 跳过图片内检测不到cat或有多个cat的图片
            if len(cat_results) != 1:
                num_skipped_photos += 1
                continue

            # 裁剪出cat
            cat_result = cat_results[0]
            crop_box = cat_result['xmin'], cat_result['ymin'], cat_result['xmax'], cat_result['ymax']

            try:
                original_image = Image.open(src_file_path).convert('RGB')
                cropped_image = original_image.crop(crop_box)

                # 保存原始裁剪后的图像
                dest_file_path = os.path.join(dest_path, f"{base_file_name}_original{ext.lower()}")
                cropped_image.save(dest_file_path, format='JPEG')

                # 对裁剪后的图像进行数据增强
                augmented_images = augment_image(cropped_image)
                for i, augmented_image in enumerate(augmented_images):
                    dest_file_path = os.path.join(dest_path, f"{base_file_name}_aug_{i}{ext.lower()}")
                    augmented_image.save(dest_file_path, format='JPEG')
                    num_augmented_photos += 1


            except Exception as e:
                print(f"处理 {file_name} 时出错: {e}")  # 更详细的错误信息
                num_skipped_photos += 1
                continue

    print(f'完成. 处理了 {num_photos - num_skipped_photos} 张照片，跳过了 {num_skipped_photos} 张照片。')
    print(f'生成了 {num_augmented_photos} 张增强后的照片。')