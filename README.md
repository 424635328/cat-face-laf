# laf版使用提示
1. 在laf函数页面添加自定义依赖`@aws-sdk/client-sts`
2. 在函数列表新建函数如下
   ![image](https://github.com/user-attachments/assets/73119e9c-d31b-4243-b8e6-ae687bf44370)

3. 复制项目根目录下的同名函数内容，粘贴到laf中创建的`get-oss-sts`函数中,测试可正常获取后发布
   ![image](https://github.com/user-attachments/assets/f09157f8-d5d8-4c1a-833b-0fb299517f97)
4. 其他步骤在进行了上面步骤后按顺序操作即可




# 中大猫谱猫脸识别

从猫谱小程序云存储拉取猫图，然后训练一个神经网络模型来识别猫猫。

## 前期准备

1. 每只猫猫积攒了超过10张高质量照片（照片清晰，且照片中只有一只猫猫）。
2. 虽然可以使用CPU进行神经网络的训练，但是速度较慢，因此最好准备一张具有足够显存的GPU。（如果显存不够，你可以根据你的显存大小适当调整图像分辨率和batch size，但是可能会导致模型训练效果下降。）
3. 网络服务器一台，用于部署识别服务。（需要自备web后端技能。）服务器不需要GPU。
4. 已注册备案域名一个，获取SSL，用于小程序调用服务。

## 手把手教程

https://docs.qq.com/doc/DSHhmTkNNYUNYTkpH

## Quick Start

### 1. 训练模型

1. 在训练服务器（或你的个人PC）上拉取本仓库代码，在仓库根目录执行`npm -v`检查node环境，执行`python --version`检查是否安装python，执行`nvcc --version`检查cuda是否正常工作，执行`conda --version`检查是否安装conda（若工具已经安装但运行以上没有正常输出，请配置环境变量，下面不在赘述）。
2. 进入`data`目录，执行`npm install`安装依赖。（需要 Node.js 环境，不确定老版本 Node.js 兼容性，建议使用最新版本。）
3. 执行`npm start`，脚本将根据小程序数据库记录拉取小程序云存储中的图片。
4. 返回仓库根目录，执行`pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu118`,安装与cuda11.8兼容的pytorch。
5. 执行`python -m pip install -r requirements.txt`安装依赖。（需要对本项目兼容性最好的Python==3.10.12和cuda==11.8。不建议使用特别新版本的 Python，可能有兼容性问题，若你已经安装conda，可以执行conda activate .conda激活已经配置好的python环境并跳过此步骤。）
6. 手动运行`prepare_yolov5.sh`中的代码行，拉取YOLOv5目标检测模型所需的代码，然后下载并预处理模型数据。
7. 执行`python3 data_preprocess.py`，脚本将使用YOLOv5从`data/photos`的图片中识别出猫猫并截取到`data/crop_photos`目录。
8. 执行`python3 main.py`，使用默认参数训练一个识别猫猫图片的模型。（你可以通过`python3 main.py --help`查看帮助来自定义一些训练参数。）程序运行结束时，你应当看到目录的export文件夹下存在`cat.onnx`和`cat.json`两个文件。（训练数据使用TensorBoard记录在`lightning_logs`文件夹下。若要查看准确率等信息，请自行运行TensorBoard。）
9. 执行`python3 main.py --data data/photos --size 224 --name fallback`，使用修改后的参数训练一个在YOLOv5无法找到猫猫时使用的全图识别模型。程序运行结束时，你应当看到目录的export文件夹下存在`fallback.onnx`和`fallback.json`两个文件。
10. 若程序异常终止提示页面文件太小，可以将虚拟内存值调大并重新执行第8步；若出现与numpy有关的报错，请执行`pip install numpy==1.23.5
`将numpy库降低版本至1.x，二次运行时更改filter值。

### 2. 部署服务
1. 在部署服务器上拉取本仓库代码。
2. 执行`pip install -r requirements.txt`安装依赖。（需要Python>=3.8。不建议使用特别新版本的 Python，可能有兼容性问题。）
3. 执行`bash prepare_yolov5.sh`拉取YOLOv5目标检测模型所需的代码，然后下载并预处理模型数据。
4. 将export文件夹从训练服务器中拷贝到部署服务器中。export文件夹中应包括四个文件：cat.onnx, cat.json, fallback.onnx, fallback.json。
5. 在仓库根目录中创建`env`文件（不需要.号开头），填写服务运行参数。示例：
    ```bash
    HOST_NAME=127.0.0.1 # 主机名
    PORT=3456 # HTTP服务端口

    SECRET_KEY='keyboard cat' # 接口密钥（用于哈希签名认证）
    TOLERANT_TIME_ERROR=60 # 调用接口时附带的时间戳参数与服务器时间之间的最大允许误差（单位：s）

    IMG_SIZE=128 # 猫猫识别模型的输入图像大小
    FALLBACK_IMG_SIZE=224 # 无法检测到猫猫时使用的全图识别模型的输入图像大小

    CAT_BOX_MAX_RET_NUM=5 # 接口返回的图片中检测到的猫猫的最大个数
    RECOGNIZE_MAX_RET_NUM=20 # 接口返回的猫猫识别结果候选列表的最大个数
    ```
6. 现在，执行`python3 app.py`，HTTP接口服务将被启动。🎉（你可以自行使用gunicorn等框架完成更规范的部署，也可以只使用pm2进行简单管理。）
