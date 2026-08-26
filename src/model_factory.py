import torch.nn as nn

from torchvision.models import (
    ResNet50_Weights,
    resnet50,
)


MODEL_NAME = "resnet50"

WEIGHTS = (
    ResNet50_Weights.IMAGENET1K_V2
)

WEIGHTS_ENUM = (
    "ResNet50_Weights.IMAGENET1K_V2"
)


def create_cloud_classifier(
    num_classes: int,
    pretrained: bool = True,
    dropout_rate: float = 0.30,
) -> nn.Module:
    """
    Membentuk ResNet50 untuk klasifikasi citra awan.

    pretrained=True:
        Menggunakan ResNet50_Weights.IMAGENET1K_V2.

    pretrained=False:
        Hanya membentuk arsitektur. Digunakan saat
        memuat checkpoint hasil training.
    """
    if num_classes < 2:
        raise ValueError(
            "Jumlah kelas minimal dua."
        )

    selected_weights = (
        WEIGHTS
        if pretrained
        else None
    )

    model = resnet50(
        weights=selected_weights
    )

    input_features = (
        model.fc.in_features
    )

    model.fc = nn.Sequential(
        nn.Dropout(
            p=dropout_rate
        ),
        nn.Linear(
            input_features,
            num_classes,
        ),
    )

    return model


def set_trainable_stage(
    model: nn.Module,
    stage: str,
) -> None:
    """
    Mengatur parameter yang dilatih.

    classifier:
        Hanya classification head.

    layer4:
        Layer residual terakhir dan classifier.

    all:
        Seluruh parameter.
    """
    model.requires_grad_(False)

    if stage == "classifier":
        model.fc.requires_grad_(True)

    elif stage == "layer4":
        model.layer4.requires_grad_(True)
        model.fc.requires_grad_(True)

    elif stage == "all":
        model.requires_grad_(True)

    else:
        raise ValueError(
            f"Tahap tidak dikenal: {stage}"
        )
        