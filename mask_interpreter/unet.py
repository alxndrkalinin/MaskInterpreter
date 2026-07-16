"""Dynamic-depth U-Net, a direct port of ``models/UNETO.get_unet`` (NCHW/NCDHW).

Fidelity notes (§1.1 of the plan — this is the main architectural risk):

- **Depth is driven solely by the first spatial axis** (``while dim[0] > 4``) while the
  strided ``k=4,s=2`` conv halves **all** axes uniformly. So ``(32,128,128)`` yields
  **3 levels**, bottleneck ``(4,16,16)`` — not "halve every axis until <=4".
- Skips are taken **before** the downsampling conv (TF order).
- 'same' padding mapping: stride-1 k3 -> pad 1; strided k4/s2 -> pad 1 (halves even
  dims); transposed k4/s2/pad1 -> exactly x2. For the used even/power-of-two patch
  sizes down/up shapes match; a defensive center-crop guards odd-dim skips.
- Final conv has **no** BatchNorm, only the configurable activation (sigmoid for the
  mask generator, linear/relu for an in-silico-labeling predictor).
- The companion ``UNet3D`` is deliberately *not* used as the template (fixed depth,
  hardcoded channel reduction, sigmoid-only head, post-downsample skips).
"""

from __future__ import annotations

import torch
from torch import nn

_ACTIVATIONS = {
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "linear": nn.Identity,
    "none": nn.Identity,
}


def _make_activation(name: str | None) -> nn.Module:
    if name is None:
        return nn.Identity()
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"unknown final activation {name!r}; choose from {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[key]()


def _center_crop_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Center-crop the spatial dims of ``x`` down to ``ref``'s (defensive, odd dims)."""
    if x.shape[2:] == ref.shape[2:]:
        return x
    slices = [slice(None), slice(None)]
    for xs, rs in zip(x.shape[2:], ref.shape[2:]):
        start = max((xs - rs) // 2, 0)
        slices.append(slice(start, start + rs))
    return x[tuple(slices)]


class UNet(nn.Module):
    """Adaptor / predictor U-Net.

    Args:
        input_size: full input shape with channels last, mirroring ``get_unet``:
            3D ``(D, H, W, C)``; 2D either ``(H, W, C)`` or ``(1, H, W, C)`` (a leading
            1 marks a 2D image, matching the TF convention).
        base_filters: initial filter count (TF ``filters = 16``).
        batch_norm: apply BatchNorm after every conv except the final (TF ``gv.batch_norm``).
        final_activation: activation on the output conv.
        out_channels: output channel count (1 for masks / single-channel predictions).
    """

    def __init__(
        self,
        input_size: tuple[int, ...],
        base_filters: int = 16,
        batch_norm: bool = True,
        final_activation: str = "sigmoid",
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        input_size = tuple(int(v) for v in input_size)
        if input_size[0] == 1 and len(input_size) == 4:  # (1, H, W, C) -> 2D
            input_size = input_size[1:]

        spatial = list(input_size[:-1])
        in_channels = input_size[-1]
        ndim = len(spatial)
        if ndim not in (2, 3):
            raise ValueError(f"input_size must be 2D or 3D spatial, got {spatial}")
        self.ndim = ndim
        self.input_size = input_size

        conv = nn.Conv3d if ndim == 3 else nn.Conv2d
        conv_t = nn.ConvTranspose3d if ndim == 3 else nn.ConvTranspose2d
        norm = (nn.BatchNorm3d if ndim == 3 else nn.BatchNorm2d) if batch_norm else None

        def cbr(cin: int, cout: int, k: int, s: int, transposed: bool = False) -> nn.Sequential:
            layer = conv_t if transposed else conv
            mods: list[nn.Module] = [layer(cin, cout, kernel_size=k, stride=s, padding=1)]
            if norm is not None:
                mods.append(norm(cout))
            mods.append(nn.ReLU())
            return nn.Sequential(*mods)

        layer_dim = list(spatial)
        filters = base_filters
        x_ch = in_channels

        self.down_pre = nn.ModuleList()   # conv1,conv2 -> skip
        self.down_sample = nn.ModuleList()  # strided downsample
        skip_channels: list[int] = []
        while layer_dim[0] > 4:
            layer_dim = [d // 2 for d in layer_dim]
            filters *= 2
            self.down_pre.append(nn.Sequential(cbr(x_ch, filters, 3, 1), cbr(filters, filters, 3, 1)))
            skip_channels.append(filters)
            self.down_sample.append(cbr(filters, filters, 4, 2))
            x_ch = filters

        # bottleneck: conv(filters*2), convt(filters)
        self.bottleneck = nn.Sequential(
            cbr(x_ch, filters * 2, 3, 1),
            cbr(filters * 2, filters, 3, 1, transposed=True),
        )
        x_ch = filters

        self.up_sample = nn.ModuleList()  # transposed upsample
        self.up_post = nn.ModuleList()    # after concat with skip
        i = len(skip_channels) - 1
        while layer_dim[0] < spatial[0]:
            f = filters
            self.up_sample.append(cbr(x_ch, f, 4, 2, transposed=True))
            layer_dim = [d * 2 for d in layer_dim]
            concat_ch = f + skip_channels[i]
            self.up_post.append(
                nn.Sequential(
                    cbr(concat_ch, f, 3, 1, transposed=True),
                    cbr(f, f, 3, 1, transposed=True),
                )
            )
            x_ch = f
            filters = filters // 2
            i -= 1

        final_conv = conv(x_ch, out_channels, kernel_size=3, stride=1, padding=1)
        self.final = nn.Sequential(final_conv, _make_activation(final_activation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for pre, down in zip(self.down_pre, self.down_sample):
            x = pre(x)
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x)
        for up, post, skip in zip(self.up_sample, self.up_post, reversed(skips)):
            x = up(x)
            x = _center_crop_to(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = post(x)
        return self.final(x)
