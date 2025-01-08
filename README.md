# <div style="text-align: center;">WildB-YOLO: A Customized YOLO Framework for Wild Berry Detection in Complex Natural Environments</div>


## Introduction 

WildB-YOLO.This model is an improved version of the latest iteration in the YOLO series, YOLOv11n, incorporating
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
