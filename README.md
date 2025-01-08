# <div style="text-align: center;">WildB-YOLO: A Customized YOLO Framework for Wild Berry Detection in Complex Natural Environments</div>


## Introduction 

WildB-YOLO.This model is an improved version of the latest iteration in the YOLO series, YOLOv11, incorporating
multiple innovative designs, including FrogFPN for multi-scale feature fusion, the SACM module for
enhanced contextual representation, and Weighted EMA Loss optimized to address class imbalance
issues. Furthermore, we employed the LAMP pruning method to achieve lightweight optimization of
the model, thereby significantly enhancing computational resource efficiency while maintaining high
detection accuracy.

## Document
### Recommended Environment

- [x] torch 3.10.14
- [x] torchvision 0.17.2+cu121

```python
pip install pypi
pip install timm==1.0.7 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.8 albumentations==1.4.11 pytorch_wavelets==1.3.0 tidecv PyWavelets opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .
```

### Train
You can choose DEYOLO's n/s/m/l/x model in [DEYOLO.yaml](./ultralytics/models/v8/DEYOLO.yaml)

```python
from ultralytics import YOLO

# Load a model
model = YOLO("ultralytics/models/v8/DEYOLO.yaml").load("yolov8n.pt")

# Train the model
train_results = model.train(
    data="M3FD.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)
```

### Predict

```python
from ultralytics import YOLO

# Load a model
model = YOLO("DEYOLOn.pt") # trained weights

# Perform object detection on RGB and IR image
model.predict([["ultralytics/assets/vi_1.png", "ultralytics/assets/ir_1.png"], # corresponding image pair
              ["ultralytics/assets/vi_2.png", "ultralytics/assets/ir_2.png"]], 
              save=True, imgsz=320, conf=0.5)
```

## Dataset
Like [M3FD.yaml](./ultralytics/yolo/cfg/M3FD.yaml) and [LLVIP.yaml](./ultralytics/yolo/cfg/LLVIP.yaml) You can use your own dataset.

<details open>
  <summary><b>File structure</b></summary>

```
Your dataset
├── ...
├── images
|   ├── vis_train
|   |   ├── 1.jpg
|   |   ├── 2.jpg
|   |   └── ...
|   ├── vis_val
|   |   ├── 1.jpg
|   |   ├── 2.jpg
|   |   └── ...
|   ├── Ir_train
|   |   ├── 100.jpg
|   |   ├── 101.jpg
|   |   └── ...
|   ├── Ir_val 
|   |   ├── 100.jpg
|   |   ├── 101.jpg
|   |   └── ...
└── labels
    ├── vis_train
    |   ├── 1.txt
    |   ├── 2.txt
    |   └── ...
    └── vis_val
        ├── 100.txt
        ├── 101.txt
        └── ...
```

</details>

You can download the dataset using the following link:
- [M3FD](https://github.com/JinyuanLiu-CV/TarDAL)
- [LLVIP](https://github.com/bupt-ai-cz/LLVIP)

## Pipeline
### The framework
<div align="center">
  <img src="imgs/network.png" alt="network" width="800" />
</div>

 We incorporate dual-context collaborative enhancement modules (DECA and DEPA) within the feature extraction
 streams dedicated to each detection head in order to refine the single-modality features
 and fuse multi-modality representations. Concurrently, the Bi-direction Decoupled Focus is inserted in the early layers of the YOLOv8 backbone to expand the network’s
 receptive fields.

### DECA and DEPA
<div align="center">
  <img src="imgs/DECA-DEPA.png" alt="DECA-DEPA" width="800" />
</div>

DECA enhances the cross-modal fusion results by leveraging dependencies between
channels within each modality and outcomes are then used to reinforce the original
single-modal features, highlighting more discriminative channels.  

DEPA is
able to learn dependency structures within and across modalities to produce enhanced
multi-modal representations with stronger positional awareness.

### Bi-direction Decoupled Focus
<div align="center">
  <img src="imgs/bi-focus.png" alt="bi-focus" width="400">
</div>

We divide the pixels into two groups for convolution.
Each group focuses on the adjacent and remote pixels at the same time.
Finally, we concatenate the original feature map in the channel dimension and
make it go through a depth-wise convolution layer.

## Visual comparison
<div align="center">
  <img src="imgs/comparison.png" alt="comparison" width="800" />
</div>

## Main Results
<div align="center">
  <img src="imgs/map.png" alt="map" width="600" />
</div>

 The mAP<sub>50</sub> and mAP<sub>50−95</sub> of every category in M<sup>3</sup>FD dataset demonstrate the superiority of our method.
 
 Trained Weights：
 - [M3FD](https://pan.baidu.com/s/1fZx0UjFcyTfRqZfgKRSZgA?pwd=3016)
 - [LLVIP](https://pan.baidu.com/s/1rw5qdCbvLTlcREoAsNMRXw?pwd=3016)


## Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```
@InProceedings{Chen_2024_ICPR,
    author    = {Chen, Yishuo and Wang, Boran and Guo, Xinyu and Zhu, Senbin and He, Jiasheng and Liu, Xiaobin and Yuan, Jing},
    title     = {DEYOLO: Dual-Feature-Enhancement YOLO for Cross-Modality Object Detection},
    booktitle = {International Conference on Pattern Recognition},
    year      = {2024},
    pages     = {}
}
```

## Acknowledgement
Part of the code is adapted from previous works: [YOLOv8](https://github.com/ultralytics/ultralytics/releases/tag/v8.1.0). We thank all the authors for their contributions.
