import torch
import torch.nn as nn
from torchvision import models

def load_backbone(backbone_name: str) -> tuple[nn.Module, int]:
    """
    Loads a backbone and freezes ALL layers except the LAST major block,
    then strips the classifier and returns (backbone_module, feature_dim).

    Backbones supported:
      resnet18/resnet34/resnet50/resnet101  → train only layer4
      inception_v3                			→ train only Mixed_7[a|b|c]
      squeezenet1_1/squeezenet1_0           → train only features.12 (Fire9)
      densenet121/densenet161/densenet169   → train only denseblock4 (+ norm5)
      vgg16/vgg19                 			→ train only Block 5 (last conv block)
    """
    name = backbone_name.lower().strip()

    def _pretrained(ctor):
        # handles both newer torchvision (weights=...) and older (pretrained=True)
        try:
            return ctor(weights="IMAGENET1K_V1")
        except Exception:
            return ctor(pretrained=True)

    # ---------------- ResNet ----------------
    if name in {"resnet18", "resnet34", "resnet50", "resnet101"}:
        ctor = {"resnet18": models.resnet18,
                "resnet34": models.resnet34,
                "resnet50": models.resnet50,
                "resnet101": models.resnet101}[name]
        m = _pretrained(ctor)

        # freeze all except layer4
        for n, p in m.named_parameters():
            if not n.startswith("layer4."):
                p.requires_grad = False

        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        # keep avgpool inside the module and flatten at the end
        backbone = nn.Sequential(
            *(c for c in [m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4]),
            m.avgpool,
            nn.Flatten(1),
        )
        return backbone, feat_dim

    # -------------- Inception v3 --------------
    if name == "inception_v3":
        m = _pretrained(models.inception_v3)
        m.aux_logits = False
        feat_dim = m.fc.in_features  # 2048 typically
        m.fc = nn.Identity()

        # freeze all except Mixed_7a/b/c
        for n, p in m.named_parameters():
            if "Mixed_7" not in n:
                p.requires_grad = False

        # Build a feature-only forward (up to Mixed_7c), then GAP + flatten
        backbone = nn.Sequential(
            m.Conv2d_1a_3x3, m.Conv2d_2a_3x3, m.Conv2d_2b_3x3, m.maxpool1,
            m.Conv2d_3b_1x1, m.Conv2d_4a_3x3, m.maxpool2,
            m.Mixed_5b, m.Mixed_5c, m.Mixed_5d,
            m.Mixed_6a, m.Mixed_6b, m.Mixed_6c, m.Mixed_6d, m.Mixed_6e,
            m.Mixed_7a, m.Mixed_7b, m.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
        )
        return backbone, feat_dim

    # -------- SqueezeNet 1.0 --------
    if name == "squeezenet1_0":
        m = _pretrained(models.squeezenet1_0)
        for n, p in m.named_parameters():
            if "features.12" not in n:
                p.requires_grad = False
                
        feat_dim = 512
        backbone = nn.Sequential(
            m.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
        )
        return backbone, feat_dim
    
    # -------- SqueezeNet 1.1 --------
    if name == "squeezenet1_1":
        m = _pretrained(models.squeezenet1_1)
        for n, p in m.named_parameters():
            if "features.12" not in n:
                p.requires_grad = False
        feat_dim = 512
        backbone = nn.Sequential(
            m.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
        )
        return backbone, feat_dim


    # -------------- DenseNet (121/161) --------------
    if name in {"densenet121", "densenet161", "densenet169"}:
        ctor = {"densenet121": models.densenet121,
                "densenet161": models.densenet161,
                "densenet169": models.densenet169}[name]
        m = _pretrained(ctor)

        # freeze all except denseblock4 and final norm5
        for n, p in m.named_parameters():
            if not n.startswith("features.denseblock4"):
                p.requires_grad = False

        feat_dim = m.classifier.in_features  # e.g., 1024 (121) / 2208 (161)
        m.classifier = nn.Identity()

        backbone = nn.Sequential(
            m.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
        )
        return backbone, feat_dim

    # -------------- VGG (16/19) --------------
    if name in {"vgg16", "vgg19"}:
        ctor = {"vgg16": models.vgg16, "vgg19": models.vgg19}[name]
        m = _pretrained(ctor)

        # Block 5 is the last conv block.
        # For vgg16, layers < 24 are blocks 1–4; for vgg19, layers < 26 are blocks 1–4.
        cutoff = 24 if name == "vgg16" else 27
        for idx, layer in m.features.named_children():
            if int(idx) < cutoff:
                for p in layer.parameters():
                    p.requires_grad = False

        # Replace avgpool to 1x1; remove classifier
        m.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        m.classifier = nn.Identity()
        feat_dim = 512  # VGG final conv channels

        backbone = nn.Sequential(
            m.features,
            m.avgpool,
            nn.Flatten(1),
        )
        return backbone, feat_dim

    raise ValueError(f"Unsupported backbone_name: {backbone_name!r}")

