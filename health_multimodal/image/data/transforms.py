#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from typing import Callable, Sequence, Optional, Tuple

import torch
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop
from torchvision import transforms
from torchvision.transforms import functional as F
import random


class ExpandChannels:
    """
    Transforms an image with one channel to an image with three channels by copying
    pixel intensities of the image along the 1st dimension.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        :param data: Tensor of shape [1, H, W].
        :return: Tensor with channel copied three times, shape [3, H, W].
        """
        if data.shape[0] != 1:
            raise ValueError(f"Expected input of shape [1, H, W], found {data.shape}")
        return torch.repeat_interleave(data, 3, dim=0)


def create_chest_xray_transform_for_inference(resize: int, center_crop_size: int) -> Compose:
    """
    Defines the image transformation pipeline for Chest-Xray datasets.

    :param resize: The size to resize the image to. Linear resampling is used.
                   Resizing is applied on the axis with smaller shape.
    :param center_crop_size: The size to center crop the image to. Square crop is applied.
    """

    transforms = [Resize(resize), CenterCrop(center_crop_size), ToTensor(), ExpandChannels()]
    return Compose(transforms)

def create_chest_xray_transform_for_train(resize: int, center_crop_size: int) -> Compose:
    
    """
    Defines the image transformation pipeline for Chest-Xray datasets.

    :param resize: The size to resize the image to. Linear resampling is used.
                   Resizing is applied on the axis with smaller shape.
    :param center_crop_size: The size to center crop the image to. Square crop is applied.
    :param color_jitter: The small color jitter to apply to the image.
    """
    transforms = [Resize(resize), CenterCrop(center_crop_size), ToTensor(), ExpandChannels(), ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)]
    return Compose(transforms)

def infer_resize_params(val_img_transforms: Sequence[Callable]) -> Tuple[Optional[int], Optional[int]]:
    """
    Given the validation transforms pipeline, extract the sizes to which the image was resized and cropped, if any.
    """
    resize_size_from_transforms = None
    crop_size_from_transforms = None
    supported_types = Resize, CenterCrop, ToTensor, ExpandChannels
    for transform in val_img_transforms:
        trsf_type = type(transform)
        if trsf_type not in supported_types:
            raise ValueError(f"Unsupported transform type {trsf_type}. Supported types are {supported_types}")
        if isinstance(transform, Resize):
            if resize_size_from_transforms is None and crop_size_from_transforms is None:
                assert transform.max_size is None
                assert isinstance(transform.size, int), f"Expected int, got {transform.size}"
                resize_size_from_transforms = transform.size
            else:
                raise ValueError("Expected Resize to be the first transform if present in val_img_transforms")
        elif isinstance(transform, CenterCrop):
            if crop_size_from_transforms is None:
                two_dims = len(transform.size) == 2
                same_sizes = transform.size[0] == transform.size[1]
                is_square = two_dims and same_sizes
                assert is_square, "Only square center crop supported"
                crop_size_from_transforms = transform.size[0]
            else:
                raise ValueError(
                    f"Crop size has already been set to {crop_size_from_transforms} in a previous transform"
                )

    return resize_size_from_transforms, crop_size_from_transforms

def get_chest_xray_transforms(size: int, crop_size: int) -> Callable:
    return ChestXrayTransforms(is_training=True, size=size, crop_size=crop_size), ChestXrayTransforms(is_training=False, size=size, crop_size=crop_size)

class ChestXrayTransforms:
    def __init__(self, is_training=True, size=224, crop_size=224):
        self.size = size
        self.is_training = is_training
        
        # Basic transforms for both training and validation
        self.basic_transforms = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            # Expand channels
            ExpandChannels(),
            # Normalize using ImageNet stats or chest X-ray specific stats if available
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Additional transforms for training only
        if is_training:
            self.train_transforms = transforms.Compose([
                # Small random rotation (chest orientation can vary slightly)
                transforms.RandomRotation(degrees=5),
                # Small random translation
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                # Very mild brightness/contrast adjustment
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
                # Random horizontal flip (anatomical structures are mostly symmetric)
                # transforms.RandomHorizontalFlip(p=0.5),
                # No vertical flip (anatomically incorrect)

            ])
    
    def __call__(self, img):
        if self.is_training:
            img = self.train_transforms(img)
        return self.basic_transforms(img)
