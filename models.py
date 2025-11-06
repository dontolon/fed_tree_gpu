import torch.nn as nn
from torchvision import models
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
)

def get_model(model_name='mobilenet_v3_small', num_classes=5, pretrained=True):
    """
    Build and return a torchvision model configured for a specific classification task.

    Args:
        model_name (str): Name of the model architecture. 
            Supported: 'mobilenet_v3_small', 'resnet18'.
        num_classes (int): Number of output classes for the final classification layer.
        pretrained (bool): If True, loads weights pretrained on ImageNet.

    Returns:
        torch.nn.Module: A model ready for training or inference, 
        with its final classification layer adjusted for `num_classes`.

    Notes:
        - For MobileNetV3, the last layer is at `model.classifier[3]`.
        - For ResNet18, the last layer is the `fc` layer.
        - Pretrained weights are automatically loaded from torchvision model weights.
        - Raises NotImplementedError if the model_name is unsupported.

    Example:
        >>> model = get_model('mobilenet_v3_small', num_classes=10, pretrained=True)
        >>> print(model.classifier[3])
        Linear(in_features=1024, out_features=10, bias=True)
    """
    if model_name == 'mobilenet_v3_small':
        # Load MobileNetV3-Small; replace the classifier head with a custom output layer
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif model_name == 'resnet18':
        # Load ResNet-18; replace the fully connected (fc) layer
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        # Raise an error for unsupported architectures
        raise NotImplementedError(f"Model '{model_name}' is not implemented yet.")

    return model
