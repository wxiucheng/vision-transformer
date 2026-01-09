<div align="center">
<h1>vision-transformer
</h1>


[Xiucheng Wang](https://wxiucheng.github.io/)&#8224; 

[Beihang University]

<a href="https://wxiucheng.github.io/">
<img src='https://img.shields.io/badge/arxiv-Cifar10Classifier-blue' alt='Paper PDF'></a>
<a href="https://wxiucheng.github.io/cifar10-classifier/">
<img src='https://img.shields.io/badge/Project-Website-orange' alt='Project Page'></a>
</div>

## 📖 Abstract
- 简单 CNN 与 ResNet18 两套模型，支持预训练微调
- 统一的 YAML 配置驱动（模型/数据/训练）
- 训练、验证、测试全流程与最优/最后权重保存
- 评估脚本与 Gradio 可视化 Demo

![识别示例](figures/dog1.png)



## 🔧 Usage

### Environment Setup

```bash
conda create -n dreamtext python=3.9
conda activate cifar10-classifier
pip install -r requirements.txt
```



### Download our Pre-trained Models

Download our available [checkpoints](https://drive.google.com/drive/folders/142L5cFw0CBoW0s2UxwODDDa38q0km0eh?usp=sharing) and put them in the corresponding directories in `./outputs/checkpoints`.



## 🚀 Gradio Demo

You can run the demo locally by
```bash
# simplecnn demo
python demo_gradio.py --cfg configs/simplecnn10_cifar10.yaml

# resnet18 demo
python demo_gradio.py --cfg configs/resnet18_cifar10.yaml 
```
<img src=figures/model_predict.png style="zoom:30%" />



## 🧪 Test & Visualization

Run evaluation and export a few sample images for visualization:
```bash
# test + save visualization images
python test.py --cfg configs/simplecnn10_cifar10.yaml

# or
python test.py --cfg configs/resnet18_cifar10.yaml
```



## 📁 Dataset

This project expects datasets under the project root `data/`:



## 💻 Training

Set the parameters in `./configs/xxx.yaml` and run:

```bash
# simplecnn
python train.py --cfg configs/simplecnn10_cifar10.yaml

# resnet18
python train.py --cfg configs/resnet18_cifar10.yaml
```



## ✨ Evaluation

Set the parameters in `./configs/test.yaml` and run:

```bash
# simplecnn
python test.py --cfg configs/simplecnn10_cifar10.yaml

# resnet18
python test.py --cfg configs/resnet18_cifar10.yaml
```
