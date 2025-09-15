import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.layers.factories import Norm
from monai.networks.nets.attentionunet import AttentionUnet
from monai.networks.nets.unet import UNet
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import (
    RandGaussianNoised,
    ScaleIntensityRanged,
)
from monai.transforms.spatial.dictionary import RandRotated, Resized
from monai.transforms.utility.dictionary import ConcatItemsd, EnsureChannelFirstd


def default_unet(
    in_channels=2,  # raw image (1) + suggested mask (1)
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    kernel_size=3,
    up_kernel_size=3,
    num_res_units=2,
    dropout=0.0,
):
    unet = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=1,  # refined mask
        channels=channels,
        strides=strides,
        kernel_size=kernel_size,
        up_kernel_size=up_kernel_size,
        num_res_units=num_res_units,
        dropout=dropout,
    )
    unet = AttentionUnet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=1,  # refined mask
        channels=channels,
        strides=strides,
        kernel_size=kernel_size,
        up_kernel_size=up_kernel_size,
    )
    # unet = nn.DataParallel(unet, device_ids=[0, 1, 3])
    unet = unet.cuda()
    return unet


# Images are in uint16
intensity_min = 0
intensity_max = 65_536

BASE_TRANSFORMS = [
    EnsureChannelFirstd(
        keys=["raw_image", "suggested_mask", "ground_truth"],
        channel_dim="no_channel",
        allow_missing_keys=True,
    ),
    ScaleIntensityRanged(
        keys=["raw_image"],
        a_min=intensity_min,
        a_max=intensity_max,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    ),
    Resized(
        keys=["raw_image", "suggested_mask", "ground_truth"],
        spatial_size=(256, 256),  # Resize to fixed size
        anti_aliasing=True,
        allow_missing_keys=True,
    ),
]

EVAL_TRANSFORMS = Compose(
    BASE_TRANSFORMS  # Combine raw image and suggested mask
    + [
        ConcatItemsd(
            keys=["raw_image", "suggested_mask"],
            name="input_combined",
            dim=0,
        )
    ],
)
TRAIN_TRANSFORMS = Compose(
    BASE_TRANSFORMS
    + [
        RandRotated(
            keys=[
                "raw_image",
                "suggested_mask",
                "ground_truth",
            ],
            range_x=(-np.radians(-20), np.radians(20)),
            allow_missing_keys=False,
            mode=["bilinear", "nearest", "nearest"],
            align_corners=[True, False, False],
        ),
        RandGaussianNoised(
            keys=["raw_image"],
            std=0.1,
            allow_missing_keys=False,
        ),
    ]
    + [  # Combine raw image and suggested mask
        ConcatItemsd(
            keys=["raw_image", "suggested_mask"],
            name="input_combined",
            dim=0,
        ),
    ]
)


WEIGHTS_FILE = "mask_refinement_unet_weights.pth"


def load_model() -> torch.nn.Module:
    """Load the default UNet model."""
    model = default_unet()
    model.load_state_dict(torch.load(WEIGHTS_FILE))
    return model


def refine_mask(
    model: torch.nn.Module,
    raw_image: torch.Tensor,
    suggested_mask: torch.Tensor,
    device: int | str | torch.device = "cuda",
) -> torch.Tensor:
    transformed = EVAL_TRANSFORMS(
        {
            "raw_image": raw_image,
            "suggested_mask": suggested_mask,
        }
    )
    input: torch.Tensor = transformed["input_combined"]  # pyright: ignore[reportArgumentType,reportCallIssue,reportAssignmentType]
    input = input.unsqueeze(0).to(device)
    with torch.no_grad():
        # Get refined mask
        refined_mask = model(input)

    # Apply sigmoid for binary segmentation
    refined_mask = torch.sigmoid(refined_mask)
    refined_mask = (refined_mask > 0.5).to(torch.uint8)  # Binarize the mask
    refined_mask = F.interpolate(refined_mask, size=raw_image.shape, mode="nearest")
    return refined_mask
