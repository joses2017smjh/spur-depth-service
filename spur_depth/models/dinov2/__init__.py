"""Vendored DINOv2 architecture (Meta Platforms, Apache-2.0 — see LICENSE).

Only the ViT architecture is vendored, not the pretrained weights. The
original code lived behind ``torch.hub.load('facebookresearch/dinov2', ...)``,
which performs a network fetch at model-construction time. That made the
model impossible to build in an offline container or a CI runner.

We do not need the network: the trained refiner checkpoint carries the full
frozen ViT-L backbone (304.4M of its 313.0M parameters sit under
``encoder.dino.*``), so the backbone is constructed with random init and then
overwritten wholesale by ``load_state_dict``.

Vendored from the torch.hub cache of ``facebookresearch/dinov2`` @ main:
  dinov2/layers/*.py
  dinov2/models/vision_transformer.py
"""

from .models import vision_transformer as vits

__all__ = ["dinov2_vitl14", "vits"]


def dinov2_vitl14(pretrained: bool = False, **kwargs):
    """Construct DINOv2 ViT-L/14, architecture-identical to the hub entrypoint.

    Keyword arguments replicate ``dinov2.hub.backbones._make_dinov2_model``
    for ``arch_name="vit_large"``, which is what
    ``torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')`` calls. The
    values are pinned here rather than defaulted so that an upstream change to
    the hub defaults cannot silently alter the architecture our checkpoint
    keys were trained against.

    Args:
        pretrained: must be False. The hub path downloads LVD-142M weights;
            we deliberately do not, because the refiner checkpoint supplies
            them. Passing True raises rather than silently reaching the
            network from inside a container.
    """
    if pretrained:
        raise ValueError(
            "pretrained=True would fetch LVD-142M weights over the network. "
            "The frozen backbone is already contained in the refiner "
            "checkpoint under encoder.dino.* — construct with pretrained=False "
            "and load the checkpoint instead."
        )

    vit_kwargs = dict(
        img_size=518,
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    )
    vit_kwargs.update(**kwargs)
    return vits.vit_large(**vit_kwargs)
