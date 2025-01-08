# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import timm
import torch
import torch.nn as nn

from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    # ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    # CBFuse,
    # CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DSConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Pose,
    RepC3,
    RepConv,
    # RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    WorldDetect,
    v10Detect,
    SpatialAttention,
    CBAM
)
from ultralytics.nn.extra_modules import *
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2EDetectLoss,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    time_sync,
    get_num_params
)

from ultralytics.nn.backbone.convnextv2 import *
from ultralytics.nn.backbone.fasternet import *
from ultralytics.nn.backbone.efficientViT import *
from ultralytics.nn.backbone.EfficientFormerV2 import *
from ultralytics.nn.backbone.VanillaNet import *
from ultralytics.nn.backbone.revcol import *
from ultralytics.nn.backbone.lsknet import *
from ultralytics.nn.backbone.SwinTransformer import *
from ultralytics.nn.backbone.repvit import *
from ultralytics.nn.backbone.CSwomTramsformer import *
from ultralytics.nn.backbone.UniRepLKNet import *
from ultralytics.nn.backbone.TransNext import *
from ultralytics.nn.backbone.rmt import *
from ultralytics.nn.backbone.pkinet import *
from ultralytics.nn.backbone.mobilenetv4 import *
from ultralytics.nn.backbone.starnet import *
from ultralytics.nn.backbone.inceptionnext import *

try:
    import thop
except ImportError:
    thop = None


DETECT_CLASS = (Detect, Detect_DyHead, Detect_AFPN_P2345, Detect_AFPN_P2345_Custom, Detect_AFPN_P345, Detect_AFPN_P345_Custom, Detect_Efficient, DetectAux, Detect_SEAM, Detect_MultiSEAM, 
                Detect_DyHeadWithDCNV3, Detect_DyHeadWithDCNV4, Detect_DyHead_Prune, Detect_LSCD, Detect_TADDH, Detect_LADH, Detect_LSCSBD, Detect_LSDECD, Detect_RSCD)
V10_DETECT_CLASS = (v10Detect,)
SEGMENT_CLASS = (Segment, Segment_Efficient, Segment_LSCD, Segment_TADDH, Segment_LADH, Segment_LSCSBD, Segment_LSDECD, Segment_RSCD)
POSE_CLASS = (Pose, Pose_LSCD, Pose_TADDH, Pose_LADH, Pose_LSCSBD, Pose_LSDECD, Pose_RSCD)
OBB_CLASS = (OBB, OBB_LSCD, OBB_TADDH, OBB_LADH, OBB_LSCSBD, OBB_LSDECD, OBB_RSCD)
C3K2_CLASS = (C3k2_Faster, C3k2_ODConv, C3k2_Faster_EMA, C3k2_DBB, C3k2_DeepDBB, C3k2_WDBB, C3k2_CloAtt, C3k2_SCConv, C3k2_ScConv, C3k2_EMSC, C3k2_EMSCP, C3k2_KW, C3k2_DySnakeConv, C3k2_DCNv2, C3k2_DCNv3,
              C3k2_OREPA, C3k2_REPVGGOREPA, C3k2_DCNv2_Dynamic, C3k2_ContextGuided, C3k2_MSBlock, C3k2_DLKA, C3k2_EMBC, C3k2_DAttention, C3k2_Parc, C3k2_DWR, C3k2_RFAConv, C3k2_RFCBAMConv, C3k2_RFCAConv,
              C3k2_FocusedLinearAttention, C3k2_MLCA, C3k2_AKConv, C3k2_UniRepLKNetBlock, C3k2_DRB, C3k2_DWR_DRB, C3k2_AggregatedAtt, C3k2_DCNv4, C3k2_SWC, C3k2_iRMB, C3k2_iRMB_Cascaded, C3k2_iRMB_DRB, C3k2_iRMB_SWC,
              C3k2_VSS, C3k2_LVMB, C3k2_DynamicConv, C3k2_GhostDynamicConv, C3k2_RVB, C3k2_RVB_SE, C3k2_RVB_EMA, C3k2_RetBlock, C3k2_PKIModule, C3k2_FADC, C3k2_PPA, C3k2_Faster_CGLU, C3k2_Star, C3k2_Star_CAA,
              C3k2_KAN, C3k2_EIEM, C3k2_DEConv, C3k2_SMPCGLU, C3k2_Heat, C3k2_WTConv, C3k2_FMB, C3k2_gConv, C3k2_AdditiveBlock, C3k2_AdditiveBlock_CGLU, C3k2_MSMHSA_CGLU, C3k2_MogaBlock, C3k2_SHSA, C3k2_SHSA_CGLU,
              C3k2_SMAFB, C3k2_SMAFB_CGLU, C3k2_IdentityFormer, C3k2_RandomMixing, C3k2_PoolingFormer, C3k2_ConvFormer, SACM, C3k2_IdentityFormerCGLU, C3k2_RandomMixingCGLU, C3k2_PoolingFormerCGLU, C3k2_ConvFormerCGLU, SACMCGLU,
              C3k2_FFCM, C3k2_MutilScaleEdgeInformationEnhance, C3k2_SFHF, C3k2_MSM, C3k2_MutilScaleEdgeInformationSelect, C3k2_HDRAB, C3k2_RAB, C3k2_LFE, C3k2_FCA_CTA, C3k2_FCA_SFA, C3k2_IDWC, C3k2_IDWB, C3k2_PConv, C3k2_EMA,
              C3k2_CAMixer
              )

class BaseModel(nn.Module):
    """The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family."""

    def forward(self, x, *args, **kwargs):
        """
        Perform forward pass of the model for either training or inference.

        If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.

        Args:
            x (torch.Tensor | dict): Input tensor for inference, or dict with image tensor and labels for training.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            (torch.Tensor): Loss if x is a dict (training), or network predictions (inference).
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            augment (bool): Augment image during prediction, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt, embeddings = [], [], []  # outputs
        for idx, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if hasattr(m, 'backbone'):
                x = m(x)
                for _ in range(5 - len(x)):
                    x.insert(0, None)
                for i_idx, i in enumerate(x):
                    if i_idx in self.save:
                        y.append(i)
                    else:
                        y.append(None)
                # print(f'layer id:{idx:>2} {m.type:>50} output shape:{", ".join([str(x_.size()) for x_ in x if x_ is not None])}')
                x = x[-1]
            else:
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
            
            # if type(x) in {list, tuple}:
            #     if idx == (len(self.model) - 1):
            #         if type(x[1]) is dict:
            #             print(f'layer id:{idx:>2} {m.type:>50} output shape:{", ".join([str(x_.size()) for x_ in x[1]["one2one"]])}')
            #         else:
            #             print(f'layer id:{idx:>2} {m.type:>50} output shape:{", ".join([str(x_.size()) for x_ in x[1]])}')
            #     else:
            #         print(f'layer id:{idx:>2} {m.type:>50} output shape:{", ".join([str(x_.size()) for x_ in x if x_ is not None])}')
            # elif type(x) is dict:
            #     print(f'layer id:{idx:>2} {m.type:>50} output shape:{", ".join([str(x_.size()) for x_ in x["one2one"]])}')
            # else:
            #     if not hasattr(m, 'backbone'):
            #         print(f'layer id:{idx:>2} {m.type:>50} output shape:{x.size()}')
            
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"WARNING ⚠️ {self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input. Appends the results to
        the provided list.

        Args:
            m (nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.

        Returns:
            None
        """
        if type(x) is tuple:
            x = list(x)
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        if type(x) is list:
            try:
                bs = x[0].size(0)
            except:
                bs = x[0][0].size(0)
        else:
            bs = x.size(0)
        o = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1E9 * 2 / bs if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {get_num_params(m):10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
                if hasattr(m, 'switch_to_deploy'):
                    m.switch_to_deploy()
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints model information.

        Args:
            detailed (bool): if True, prints out detailed information about the model. Defaults to False
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        Applies a function to all the tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): the function to apply to the model

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, DETECT_CLASS):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        if verbose:
            LOGGER.info(f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


class DetectionModel(BaseModel):
    """YOLOv8 detection model."""

    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "WARNING ⚠️ YOLOv9 `Silence` module is deprecated in favor of nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Warehouse_Manager
        warehouse_manager_flag = self.yaml.get('Warehouse_Manager', False)
        self.warehouse_manager = None
        if warehouse_manager_flag:
            self.warehouse_manager = Warehouse_Manager(cell_num_ratio=self.yaml.get('Warehouse_Manager_Ratio', 1.0))
        
        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose, warehouse_manager=self.warehouse_manager)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        if warehouse_manager_flag:
            self.warehouse_manager.store()
            self.warehouse_manager.allocate(self)
            self.net_update_temperature(0)
        
        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (DETECT_CLASS + SEGMENT_CLASS + POSE_CLASS + OBB_CLASS + V10_DETECT_CLASS)):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            s = 640  # 2x min stride
            m.inplace = self.inplace

            def _forward(x):
                """Performs a forward pass through the model, handling different Detect subclass types accordingly."""
                if isinstance(m, (DetectAux,)):
                    return self.forward(x)[:3]
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)[0] if isinstance(m, SEGMENT_CLASS + POSE_CLASS + OBB_CLASS) else self.forward(x)

            try:
                m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(2, ch, s, s))])  # forward
            except RuntimeError as e:
                if 'Not implemented on the CPU' in str(e) or 'Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)' in str(e) or \
                'CUDA tensor' in str(e) or 'is_cuda()' in str(e) or 'carafe_forward_impl' in str(e):
                    self.model.to(torch.device('cuda'))
                    m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(2, ch, s, s).to(torch.device('cuda')))])  # forward
                else:
                    raise e
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("WARNING ⚠️ Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)

    def net_update_temperature(self, temp):
        for m in self.modules():
            if hasattr(m, "update_temperature"):
                m.update_temperature(temp)

class OBBModel(DetectionModel):
    """YOLOv8 Oriented Bounding Box (OBB) model."""

    def __init__(self, cfg="yolov8n-obb.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 OBB model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the model."""
        return v8OBBLoss(self)


class SegmentationModel(DetectionModel):
    """YOLOv8 segmentation model."""

    def __init__(self, cfg="yolov8n-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 segmentation model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the SegmentationModel."""
        return v8SegmentationLoss(self)


class PoseModel(DetectionModel):
    """YOLOv8 pose model."""

    def __init__(self, cfg="yolov8n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize YOLOv8 Pose model."""
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""
        return v8PoseLoss(self)


class ClassificationModel(BaseModel):
    """YOLOv8 classification model."""

    def __init__(self, cfg="yolov8n-cls.yaml", ch=3, nc=None, verbose=True):
        """Init ClassificationModel with YAML, channels, number of classes, verbose flag."""
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set YOLOv8 model configurations and define the model architecture."""
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """Update a TorchVision classification model to class count 'n' if required."""
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, nn.Linear(m.in_features, nc))
        elif isinstance(m, nn.Sequential):
            types = [type(x) for x in m]
            if nn.Linear in types:
                i = len(types) - 1 - types[::-1].index(nn.Linear)  # last nn.Linear index
                if m[i].out_features != nc:
                    m[i] = nn.Linear(m[i].in_features, nc)
            elif nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(nn.Conv2d)  # last nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)

    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """
    RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.

    Attributes:
        cfg (str): The configuration file path or preset string. Default is 'rtdetr-l.yaml'.
        ch (int): Number of input channels. Default is 3 (RGB).
        nc (int, optional): Number of classes for object detection. Default is None.
        verbose (bool): Specifies if summary statistics are shown during initialization. Default is True.

    Methods:
        init_criterion: Initializes the criterion used for loss calculation.
        loss: Computes and returns the loss during training.
        predict: Performs a forward pass through the network and returns the output.
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize the RTDETRDetectionModel.

        Args:
            cfg (str): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes. Defaults to None.
            verbose (bool, optional): Print additional information during initialization. Defaults to True.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """
        Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (torch.Tensor, optional): Precomputed model predictions. Defaults to None.

        Returns:
            (tuple): A tuple containing the total loss and main three losses in a tensor.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            batch (dict, optional): Ground truth data for evaluation. Defaults to None.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x


class WorldModel(DetectionModel):
    """YOLOv8 World Model."""

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 world model with given config and parameters."""
        self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder
        self.clip_model = None  # CLIP model placeholder
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """Set classes in advance so that model could do offline-inference without clip model."""
        try:
            import clip
        except ImportError:
            check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip

        if (
            not getattr(self, "clip_model", None) and cache_clip_model
        ):  # for backwards compatibility of models lacking clip_model attribute
            self.clip_model = clip.load("ViT-B/32")[0]
        model = self.clip_model if cache_clip_model else clip.load("ViT-B/32")[0]
        device = next(model.parameters()).device
        text_token = clip.tokenize(text).to(device)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        self.txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        self.model[-1].nc = len(text)

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            txt_feats (torch.Tensor): The text features, use it if it's given. Defaults to None.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if len(txt_feats) != len(x):
            txt_feats = txt_feats.repeat(len(x), 1, 1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLO network's final layer."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.

    Example:
        ```python
        with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
            import old.module  # this will now import new.module
            from old.module import attribute  # this will now import new.module.attribute
        ```

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # Set attributes in sys.modules under their old name
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class SafeClass:
    """A placeholder class to replace unknown classes during unpickling."""

    def __init__(self, *args, **kwargs):
        """Initialize SafeClass instance, ignoring all arguments."""
        pass

    def __call__(self, *args, **kwargs):
        """Run SafeClass instance, ignoring all arguments."""
        pass


class SafeUnpickler(pickle.Unpickler):
    """Custom Unpickler that replaces unknown classes with SafeClass."""

    def find_class(self, module, name):
        """Attempt to find a class, returning SafeClass if not among safe modules."""
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # Add other modules considered safe
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass


def torch_safe_load(weight, safe_only=False):
    """
    Attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches the
    error, logs a warning message, and attempts to install the missing module via the check_requirements() function.
    After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.
        safe_only (bool): If True, replace unknown classes with SafeClass during loading.

    Example:
    ```python
    from ultralytics.nn.tasks import torch_safe_load

    ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    ```

    Returns:
        ckpt (dict): The loaded model checkpoint.
        file (str): The loaded filename
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
            },
        ):
            if safe_only:
                # Load via custom pickle module
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch.load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch.load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolov8n.pt'"
                )
            ) from e
        LOGGER.warning(
            f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolov8n.pt'"
        )
        check_requirements(e.name)  # install missing module
        ckpt = torch.load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"WARNING ⚠️ The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""
    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # load ckpt
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None  # combined args
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = guess_model_task(model)
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        # Append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())  # model in eval mode

    # Module updates
    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    LOGGER.info(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = guess_model_task(model)
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True, warehouse_manager=None):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        if len(scales[scale]) == 3:
            depth, width, max_channels = scales[scale]
        elif len(scales[scale]) == 4:
            depth, width, max_channels, threshold = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    is_backbone = False
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        try:
            if m == 'node_mode':
                m = d[m]
                if len(args) > 0:
                    if args[0] == 'head_channel':
                        args[0] = int(d[args[0]])
            t = m
            m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        except:
            pass
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    try:
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                    except:
                        args[j] = a

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in ((
            Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, C2fPSA, C2PSA, DWConv, Focus, BottleneckCSP,
            C1, C2, C2f, C3k2, RepNCSPELAN4, ELAN1, ADown, AConv, SPPELAN, C2fAttn, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d,
            C3x, RepC3, PSA, SCDown, C2fCIB, GSConv, GSConvns, VoVGSCSP, VoVGSCSPns, VoVGSCSPC, RCSOSA, KWConv, CSPStage, SPDConv, RepBlock,
            SPPF_LSKA, CSP_EDLAN, nn.Conv2d, HWD, RepNCSPELAN4, DBBNCSPELAN4, OREPANCSPELAN4, DRBNCSPELAN4, RepNCSPELAN4_CAA, V7DownSampling,
            DGCST, RepNCSPELAN4_CAA, SRFD, DRFD, RGCSPELAN, CSP_PTB, SimpleStem, VisionClueMerge, VSSBlock_YOLO, XSSBlock, GLSA, FeaturePyramidSharedConv,
            LDConv, CSP_MSCB, CSP_PMSFA, RFAConv, RFCBAMConv, RFCAConv, CSP_FreqSpatial, C2BRA, C2CGA, C2DA, C2DPB, MANet
        ) + C3K2_CLASS):
            if args[0] == 'head_channel':
                args[0] = d[args[0]]
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)  # embed channels
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                )  # num heads

            args = [c1, c2, *args[1:]]
            if m in (KWConv, C3k2_KW):
                args.insert(2, f'layer{i}')
                args.insert(2, warehouse_manager)
            if m in (DySnakeConv,):
                c2 = c2 * 3
            if m in (RepNCSPELAN4, DBBNCSPELAN4, OREPANCSPELAN4, DRBNCSPELAN4, RepNCSPELAN4_CAA):
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
                args[3] = make_divisible(min(args[3], max_channels) * width, 8)
            if m in ((
                BottleneckCSP, C1, C2, C2f, C3k2, C2fAttn, C3, C3TR, C3Ghost, C3x, RepC3, C2fPSA, C2fCIB, C2PSA, VoVGSCSP, VoVGSCSPns, VoVGSCSPC, RCSOSA,
                CSPStage, SPDConv, RepBlock, CSP_EDLAN, RGCSPELAN, CSP_PTB, XSSBlock, CSP_MSCB, CSP_PMSFA, CSP_FreqSpatial, C2BRA, C2CGA, C2DA, C2DPB, MANet
            ) + C3K2_CLASS):
                args.insert(2, n)  # number of repeats
                n = 1
            if m in ((C3k2,) + C3K2_CLASS) and scale in "mlx":  # for M/L/X sizes
                if m in (C3k2_AggregatedAtt, C3k2_MSBlock, C3k2_DAttention, C3k2_FocusedLinearAttention, C3k2_Parc, C3k2_UniRepLKNetBlock, C3k2_SMPCGLU, 
                         C3k2_Heat, C3k2_RandomMixing, C3k2_RandomMixingCGLU, C3k2_PKIModule, C3k2_FCA_CTA):
                    if type(args[-1]) == bool:
                        args[-1] = True
                    else:
                        args[-2] = True
                else:
                    args[3] = True
        elif m in {AIFI, AIFI_RepBN}:
            args = [ch[f], *args]
            c2 = args[0]
        elif m in (HGStem, HGBlock, Ghost_HGBlock, Rep_HGBlock, Dynamic_HGBlock, EIEStem):
            c1, cm, c2 = ch[f], args[0], args[1]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
                cm = make_divisible(min(cm, max_channels) * width, 8)
            args = [c1, cm, c2, *args[2:]]
            if m in (HGBlock, Ghost_HGBlock, Rep_HGBlock, Dynamic_HGBlock):
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in ((WorldDetect, ImagePoolingAttn) + DETECT_CLASS + V10_DETECT_CLASS + SEGMENT_CLASS + POSE_CLASS + OBB_CLASS):
            args.append([ch[x] for x in f])
            if m in SEGMENT_CLASS:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
                if m in (Segment_LSCD, Segment_TADDH, Segment_LSCSBD, Segment_LSDECD, Segment_RSCD):
                    args[3] = make_divisible(min(args[3], max_channels) * width, 8)
            if m in (Detect_LSCD, Detect_TADDH, Detect_LSCSBD, Detect_LSDECD, Detect_RSCD):
                args[1] = make_divisible(min(args[1], max_channels) * width, 8)
            if m in (Pose_LSCD, Pose_TADDH, Pose_LSCSBD, Pose_LSDECD, Pose_RSCD, OBB_LSCD, OBB_TADDH, OBB_LSCSBD, OBB_LSDECD, OBB_RSCD):
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is Fusion:
            args[0] = d[args[0]]
            c1, c2 = [ch[x] for x in f], (sum([ch[x] for x in f]) if args[0] == 'concat' else ch[f[0]])
            args = [c1, args[0]]
        elif m is CBLinear:
            c2 = make_divisible(min(args[0][-1], max_channels) * width, 8)
            c1 = ch[f]
            args = [c1, [make_divisible(min(c2_, max_channels) * width, 8) for c2_ in args[0]], *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif isinstance(m, str):
            t = m
            if len(args) == 2:        
                m = timm.create_model(m, pretrained=args[0], pretrained_cfg_overlay={'file':args[1]}, features_only=True)
            elif len(args) == 1:
                m = timm.create_model(m, pretrained=args[0], features_only=True)
            c2 = m.feature_info.channels()
        elif m in {convnextv2_atto, convnextv2_femto, convnextv2_pico, convnextv2_nano, convnextv2_tiny, convnextv2_base, convnextv2_large, convnextv2_huge,
                   fasternet_t0, fasternet_t1, fasternet_t2, fasternet_s, fasternet_m, fasternet_l,
                   EfficientViT_M0, EfficientViT_M1, EfficientViT_M2, EfficientViT_M3, EfficientViT_M4, EfficientViT_M5,
                   efficientformerv2_s0, efficientformerv2_s1, efficientformerv2_s2, efficientformerv2_l,
                   vanillanet_5, vanillanet_6, vanillanet_7, vanillanet_8, vanillanet_9, vanillanet_10, vanillanet_11, vanillanet_12, vanillanet_13, vanillanet_13_x1_5, vanillanet_13_x1_5_ada_pool,
                   RevCol,
                   lsknet_t, lsknet_s,
                   SwinTransformer_Tiny,
                   repvit_m0_9, repvit_m1_0, repvit_m1_1, repvit_m1_5, repvit_m2_3,
                   CSWin_tiny, CSWin_small, CSWin_base, CSWin_large,
                   unireplknet_a, unireplknet_f, unireplknet_p, unireplknet_n, unireplknet_t, unireplknet_s, unireplknet_b, unireplknet_l, unireplknet_xl,
                   transnext_micro, transnext_tiny, transnext_small, transnext_base,
                   RMT_T, RMT_S, RMT_B, RMT_L,
                   PKINET_T, PKINET_S, PKINET_B,
                   MobileNetV4ConvSmall, MobileNetV4ConvMedium, MobileNetV4ConvLarge, MobileNetV4HybridMedium, MobileNetV4HybridLarge,
                   starnet_s050, starnet_s100, starnet_s150, starnet_s1, starnet_s2, starnet_s3, starnet_s4,
                   inceptionnext_tiny, inceptionnext_small, inceptionnext_base, inceptionnext_base_384,
                   }:
            if m is RevCol:
                args[1] = [make_divisible(min(k, max_channels) * width, 8) for k in args[1]]
                args[2] = [max(round(k * depth), 1) for k in args[2]]
            m = m(*args)
            c2 = m.channel
        elif m in {EMA, SpatialAttention, BiLevelRoutingAttention, BiLevelRoutingAttention_nchw,
                   TripletAttention, CoordAtt, CBAM, BAMBlock, LSKBlock, ScConv, LAWDS, EMSConv, EMSConvP,
                   SEAttention, CPCA, Partial_conv3, FocalModulation, EfficientAttention, MPCA, deformable_LKA,
                   EffectiveSEModule, LSKA, SegNext_Attention, DAttention, MLCA, TransNeXt_AggregatedAttention,
                   FocusedLinearAttention, LocalWindowAttention, ChannelAttention_HSFPN, ELA_HSFPN, CA_HSFPN, CAA_HSFPN, 
                   DySample, CARAFE, CAA, ELA, CAFM, AFGCAttention, EUCB, EfficientChannelAttention}:
            c2 = ch[f]
            args = [c2, *args]
            # print(args)
        elif m in {SimAM, SpatialGroupEnhance}:
            c2 = ch[f]
            args = [*args]
        elif m is ContextGuidedBlock_Down:
            c2 = ch[f] * 2
            args = [ch[f], c2, *args]
        elif m is BiFusion:
            c1 = [ch[x] for x in f]
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            args = [c1, c2]
        # --------------GOLD-YOLO--------------
        elif m in {SimFusion_4in, AdvPoolFusion}:
            c2 = sum(ch[x] for x in f)
        elif m is SimFusion_3in:
            c2 = args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [[ch[f_] for f_ in f], c2]
        elif m is IFM:
            c1 = ch[f]
            c2 = sum(args[0])
            args = [c1, *args]
        elif m is InjectionMultiSum_Auto_pool:
            c1 = ch[f[0]]
            c2 = args[0]
            args = [c1, *args]
        elif m is PyramidPoolAgg:
            c2 = args[0]
            args = [sum([ch[f_] for f_ in f]), *args]
        elif m is TopBasicLayer:
            c2 = sum(args[1])
        # --------------GOLD-YOLO--------------
        # --------------ASF--------------
        elif m is Zoom_cat:
            c2 = sum(ch[x] for x in f)
        elif m is Add:
            c2 = ch[f[-1]]
        elif m in {ScalSeq, DynamicScalSeq}:
            c1 = [ch[x] for x in f]
            c2 = make_divisible(args[0] * width, 8)
            args = [c1, c2]
        elif m is asf_attention_model:
            args = [ch[f[-1]]]
        # --------------ASF--------------
        elif m is SDI:
            args = [[ch[x] for x in f]]
        elif m is Multiply:
            c2 = ch[f[0]]
        elif m is FocusFeature:
            c1 = [ch[x] for x in f]
            c2 = int(c1[1] * 0.5 * 3)
            args = [c1, *args]
        elif m is SAFI:
            c1 = [ch[x] for x in f]
            args = [c1, c2]
        elif m is CSMHSA:
            c1 = [ch[x] for x in f]
            c2 = ch[f[-1]]
            args = [c1, c2]
        elif m is CFC_CRB:
            c1 = ch[f]
            c2 = c1 // 2
            args = [c1, *args]
        elif m is SFC_G2:
            c1 = [ch[x] for x in f]
            c2 = c1[0]
            args = [c1]
        elif m in {CGAFusion, CAFMFusion, SDFM, PSFM}:
            c2 = ch[f[1]]
            args = [c2, *args]
        elif m in {ContextGuideFusionModule}:
            c1 = [ch[x] for x in f]
            c2 = 2 * c1[1]
            args = [c1]
        # elif m in {PSA}:
        #     c2 = ch[f]
        #     args = [c2, *args]
        elif m in {SBA}:
            c1 = [ch[x] for x in f]
            c2 = c1[-1]
            args = [c1, c2]
        elif m in {WaveletPool}:
            c2 = ch[f] * 4
        elif m in {WaveletUnPool}:
            c2 = ch[f] // 4
        elif m in {CSPOmniKernel}:
            c2 = ch[f]
            args = [c2]
        elif m in {ChannelTransformer, PyramidContextExtraction}:
            c1 = [ch[x] for x in f]
            c2 = c1
            args = [c1]
        elif m in {RCM}:
            c2 = ch[f]
            args = [c2, *args]
        elif m in {DynamicInterpolationFusion}:
            c2 = ch[f[0]]
            args = [[ch[x] for x in f]]
        elif m in {FuseBlockMulti}:
            c2 = ch[f[0]]
            args = [c2]
        elif m in {CrossLayerChannelAttention, CrossLayerSpatialAttention}:
            c2 = [ch[x] for x in f]
            args = [c2[0], *args]
        elif m in {FreqFusion}:
            c2 = ch[f[0]]
            args = [[ch[x] for x in f], *args]
        elif m in {DynamicAlignFusion}:
            c2 = args[0]
            args = [[ch[x] for x in f], c2]
        elif m in {ConvEdgeFusion}:
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            args = [[ch[x] for x in f], c2]
        elif m in {MutilScaleEdgeInfoGenetator}:
            c1 = ch[f]
            c2 = [make_divisible(min(i, max_channels) * width, 8) for i in args[0]]
            args = [c1, c2]
        elif m is HyperComputeModule:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, threshold]
        elif m in {GetIndexOutput}:
            c2 = ch[f][args[0]]
        else:
            c2 = ch[f]

        if isinstance(c2, list) and m not in {ChannelTransformer, PyramidContextExtraction, CrossLayerChannelAttention, CrossLayerSpatialAttention, MutilScaleEdgeInfoGenetator}:
            is_backbone = True
            m_ = m
            m_.backbone = True
        else:
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
            t = str(m)[8:-2].replace('__main__.', '')  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i + 4 if is_backbone else i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(x % (i + 4 if is_backbone else i) for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        if isinstance(c2, list) and m not in {ChannelTransformer, PyramidContextExtraction, CrossLayerChannelAttention, CrossLayerSpatialAttention, MutilScaleEdgeInfoGenetator}:
            ch.extend(c2)
            for _ in range(5 - len(ch)):
                ch.insert(0, 0)
        else:
            ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    unified_path = re.sub(r'(hyper-yolo)([nslmx])(.+)?$', r'\1\3', str(unified_path))
    unified_path = re.sub(r'(hyper-yolot)(.+)?$', r'\1\2', str(unified_path))
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale. The function
    uses regular expression matching to find the pattern of the model scale in the YAML file name, which is denoted by
    n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    try:
        if re.search(r'(hyper-yolo)([tnslmx])', Path(model_path).stem) is not None:
            return re.search(r'(hyper-yolo)([tnslmx])', Path(model_path).stem).group(2)
        else:
            return re.search(r"yolo[v]?\d+([nslmx])", Path(model_path).stem).group(1)  # n, s, m, l, or x
    except AttributeError:
        return ""


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if "segment" in m:
            return "segment"
        if "pose" in m:
            return "pose"
        if "obb" in m:
            return "obb"

    # Guess from model cfg
    if isinstance(model, dict):
        try:
            return cfg2task(model)
        except:  # noqa E722
            pass

    # Guess from PyTorch model
    if isinstance(model, nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            try:
                return eval(x)["task"]
            except:  # noqa E722
                pass
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            try:
                return cfg2task(eval(x))
            except:  # noqa E722
                pass

        for m in model.modules():
            if isinstance(m, Segment):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, v10Detect)):
                return "detect"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    LOGGER.warning(
        "WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect
