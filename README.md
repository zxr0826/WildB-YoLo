# <div style="text-align: center;">WildB-YOLO: A Customized YOLO Framework for Wild Berry Detection in Complex Natural Environments</div>


## Introduction 

WildB-YOLO
WildB-YOLO is an advanced iteration of the YOLOv11n model, featuring several cutting-edge enhancements designed to push the boundaries of object detection. Key innovations include the FrogFPN for multi-scale feature fusion, the SACM module for richer contextual representations, and the Weighted EMA Loss, which effectively addresses class imbalance challenges. Additionally, we employ the LAMP pruning technique to streamline the model, achieving lightweight optimization without compromising detection accuracy. This combination results in a highly efficient model that significantly reduces computational resource demands while maintaining exceptional performance.

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
You can choose WildB-YoLo's model in [WildB-YoLo.yaml](./ultralytics/cfg/models/11/WildB-YoLo.yaml)

```python
from ultralytics import YOLO

# Load a model
model = YOLO("ultralytics/cfg/models/11/WildB-YoLo.yaml")

# Train the model
model.train(
                data='data.yaml', # pase your dataset
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=0,
                workers=4, 
                optimizer='SGD', 
                project='runs/train',
                name='exp',
)
