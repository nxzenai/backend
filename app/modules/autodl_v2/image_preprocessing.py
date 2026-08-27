from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode


LEGACY_IMAGE_PREPROCESSING_VERSION = "autodl_v2_rgb_imagenet_v1"
IMAGE_PREPROCESSING_VERSION = "autodl_v2_rgb_letterbox_v2"


class _AspectRatioResizePad:
    def __init__(self, size: int, fill: tuple[int, int, int]):
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        contained = ImageOps.contain(
            image, (self.size, self.size), method=Image.Resampling.BILINEAR,
        )
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        offset = ((self.size - contained.width) // 2, (self.size - contained.height) // 2)
        canvas.paste(contained, offset)
        return canvas


@dataclass
class ImageInference:
    tensor: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_indices: torch.Tensor
    predicted_labels: list[str]


def validate_image_preprocessing(metadata: dict[str, Any]) -> None:
    version = metadata.get("transform_version")
    if version not in {None, LEGACY_IMAGE_PREPROCESSING_VERSION, IMAGE_PREPROCESSING_VERSION}:
        raise ValueError("The saved image preprocessing version is not supported.")
    if metadata.get("kind") != "image" or metadata.get("color_mode") != "RGB":
        raise ValueError("The saved image color preprocessing is inconsistent.")
    try:
        channels = int(metadata.get("channels", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("The saved image channel configuration is invalid.") from exc
    if channels != 3:
        raise ValueError("The saved image channel configuration is inconsistent.")
    image_size = metadata.get("image_size")
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("The saved image size is invalid.")
    mean = metadata.get("normalization_mean")
    std = metadata.get("normalization_std")
    if not isinstance(mean, list) or not isinstance(std, list) or len(mean) != 3 or len(std) != 3:
        raise ValueError("The saved image normalization metadata is incomplete.")
    try:
        numeric_mean = [float(value) for value in mean]
        numeric_std = [float(value) for value in std]
        if not all(math.isfinite(value) for value in [*numeric_mean, *numeric_std]):
            raise ValueError("The saved image normalization values must be finite.")
        if any(value <= 0 for value in numeric_std):
            raise ValueError("The saved image normalization scale is invalid.")
    except (TypeError, ValueError) as exc:
        raise ValueError("The saved image normalization metadata is invalid.") from exc
    if metadata.get("resize_interpolation", "bilinear") != "bilinear":
        raise ValueError("The saved image resize interpolation is not supported.")
    if metadata.get("resize_antialias", True) is not True:
        raise ValueError("The saved image resize antialias setting is not supported.")
    if version == IMAGE_PREPROCESSING_VERSION:
        if metadata.get("resize_strategy") != "aspect_ratio_pad":
            raise ValueError("The saved image resize strategy is inconsistent.")
        padding = metadata.get("padding_color")
        if not isinstance(padding, list) or len(padding) != 3:
            raise ValueError("The saved image padding configuration is invalid.")
        if metadata.get("exif_transpose") is not True or metadata.get("alpha_background") != "padding_color":
            raise ValueError("The saved image orientation or alpha handling is inconsistent.")


def build_image_transform(
    metadata: dict[str, Any], *, augmentation: dict[str, Any] | None = None,
) -> transforms.Compose:
    """Build the single V2 image tensor pipeline used by train, validation, and inference."""
    validate_image_preprocessing(metadata)
    if metadata.get("transform_version") == IMAGE_PREPROCESSING_VERSION:
        steps: list[Any] = [
            _AspectRatioResizePad(
                metadata["image_size"], tuple(int(value) for value in metadata["padding_color"]),
            ),
        ]
    else:
        steps = [
            transforms.Resize(
                (metadata["image_size"], metadata["image_size"]),
                interpolation=InterpolationMode.BILINEAR, antialias=True,
            ),
        ]
    if augmentation and augmentation.get("enabled"):
        steps.extend([
            transforms.RandomAffine(
                degrees=float(augmentation["rotation_degrees"]),
                translate=tuple(augmentation["translation_fraction"]),
                scale=tuple(augmentation["scale_range"]),
                interpolation=InterpolationMode.BILINEAR,
                fill=tuple(int(value) for value in metadata.get("padding_color", [255, 255, 255])),
            ),
            transforms.ColorJitter(
                brightness=float(augmentation["brightness"]),
                contrast=float(augmentation["contrast"]),
            ),
        ])
        if augmentation.get("horizontal_flip"):
            steps.append(transforms.RandomHorizontalFlip())
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(metadata["normalization_mean"], metadata["normalization_std"]),
    ])
    return transforms.Compose(steps)


def image_to_tensor(
    image: Image.Image, metadata: dict[str, Any], *,
    augmentation: dict[str, Any] | None = None,
) -> torch.Tensor:
    prepared = prepare_pil_image(image, metadata)
    return build_image_transform(metadata, augmentation=augmentation)(prepared)


def prepare_pil_image(image: Image.Image, metadata: dict[str, Any]) -> Image.Image:
    if metadata.get("transform_version") != IMAGE_PREPROCESSING_VERSION:
        return image.convert("RGB")
    oriented = ImageOps.exif_transpose(image)
    fill = tuple(int(value) for value in metadata["padding_color"])
    if "A" in oriented.getbands() or (oriented.mode == "P" and "transparency" in oriented.info):
        rgba = oriented.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (*fill, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return oriented.convert("RGB")


def run_image_inference(
    model: torch.nn.Module, *, classes: list[str], device: torch.device,
    image: Image.Image | None = None, tensor: torch.Tensor | None = None,
    preprocessing: dict[str, Any] | None = None,
) -> ImageInference:
    """Canonical deterministic V2 image inference path for validation and prediction."""
    if (image is None) == (tensor is None):
        raise ValueError("Provide exactly one image or prepared image tensor for inference.")
    if image is not None:
        if preprocessing is None:
            raise ValueError("Saved image preprocessing metadata is required for inference.")
        tensor = image_to_tensor(image, preprocessing)
    assert tensor is not None
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError("The prepared image tensor has an invalid shape.")
    model.eval()
    device_tensor = tensor.to(device)
    with torch.inference_mode():
        logits = model(device_tensor)
        if logits.ndim != 2 or logits.shape[1] != len(classes):
            raise ValueError("The image model output does not match the saved class order.")
        probabilities = torch.softmax(logits, dim=1)
        predicted_indices = torch.argmax(probabilities, dim=1)
    cpu_tensor = device_tensor.detach().cpu()
    cpu_logits = logits.detach().cpu()
    cpu_probabilities = probabilities.detach().cpu()
    cpu_indices = predicted_indices.detach().cpu()
    return ImageInference(
        tensor=cpu_tensor,
        logits=cpu_logits,
        probabilities=cpu_probabilities,
        predicted_indices=cpu_indices,
        predicted_labels=[classes[int(index)] for index in cpu_indices.tolist()],
    )


__all__ = [
    "IMAGE_PREPROCESSING_VERSION", "LEGACY_IMAGE_PREPROCESSING_VERSION",
    "ImageInference", "build_image_transform", "image_to_tensor", "prepare_pil_image",
    "run_image_inference", "validate_image_preprocessing",
]
