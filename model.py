from .edgenext import EdgeNeXt
from .edgenext_bn_hs import EdgeNeXtBNHS
from timm.models.registry import register_model

# CIFAR-10 specific configurations for EdgeNeXt models

@register_model
def edgenext_xx_small_cifar10(pretrained=False, **kwargs):
    model = EdgeNeXt(depths=[2, 2, 6, 2], dims=[24, 48, 88, 168], expan_ratio=4,
                     global_block=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False, False],
                     kernel_sizes=[3, 5, 7, 9],
                     heads=[4, 4, 4, 4],
                     d2_scales=[2, 2, 3, 4],
                     num_classes=10,  # CIFAR-10 classes
                     **kwargs)
    return model


@register_model
def edgenext_x_small_cifar10(pretrained=False, **kwargs):
    model = EdgeNeXt(depths=[3, 3, 9, 3], dims=[32, 64, 100, 192], expan_ratio=4,
                     global_block=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False, False],
                     kernel_sizes=[3, 5, 7, 9],
                     heads=[4, 4, 4, 4],
                     d2_scales=[2, 2, 3, 4],
                     num_classes=10,  # CIFAR-10 classes
                     **kwargs)
    return model


@register_model
def edgenext_small_cifar10(pretrained=False, **kwargs):
    model = EdgeNeXt(depths=[3, 3, 9, 3], dims=[48, 96, 160, 304], expan_ratio=4,
                     global_block=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False, False],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 2, 3, 4],
                     num_classes=10,  # CIFAR-10 classes
                     **kwargs)
    return model


@register_model
def edgenext_base_cifar10(pretrained=False, **kwargs):
    model = EdgeNeXt(depths=[3, 3, 9, 3], dims=[80, 160, 288, 584], expan_ratio=4,
                     global_block=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False, False],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 2, 3, 4],
                     num_classes=10,  # CIFAR-10 classes
                     **kwargs)
    return model


# Models with Batch Normalization and Hard-Swish Activation

@register_model
def edgenext_xx_small_bn_hs_cifar10(pretrained=False, **kwargs):
    model = EdgeNeXtBNHS(depths=[2, 2, 6, 2], dims=[24, 48, 88, 168], expan_ratio=4,
                         global_block=[0, 1, 1, 1],
                         global_block_type=['None', 'SDTA_BN_HS', 'SDTA_BN_HS', 'SDTA_BN_HS'],
                         use_pos_embd_xca=[False, True, False, False],
                         kernel_sizes=[3, 5, 7, 9],
                         heads=[4, 4, 4, 4],
                         d2_scales=[2, 2, 3, 4],
                         num_classes=10,  # CIFAR-10 classes
                         **kwargs)
    return model


@register_model
def edgenext_x_small_bn_hs_cifar10(pretrained=False, **kwargs):
    model = EdgeNeXtBNHS(depths=[3, 3, 9, 3], dims=[32, 64, 100, 192], expan_ratio=4,
                         global_block=[0, 1, 1, 1],
                         global_block_type=['None', 'SDTA_BN_HS', 'SDTA_BN_HS', 'SDTA_BN_HS'],
                         use_pos_embd_xca=[False, True, False, False],
                         kernel_sizes=[3, 5, 7, 9],
                         heads=[4, 4, 4, 4],
                         d2_scales=[2, 2, 3, 4],
                         num_classes=10,  # CIFAR-10 classes
                         **kwargs)
    return model


@register_model
def edgenext_small_bn_hs_cifar10(pretrained=False, **kwargs):
    model = EdgeNeXtBNHS(depths=[3, 3, 9, 3], dims=[48, 96, 160, 304], expan_ratio=4,
                         global_block=[0, 1, 1, 1],
                         global_block_type=['None', 'SDTA_BN_HS', 'SDTA_BN_HS', 'SDTA_BN_HS'],
                         use_pos_embd_xca=[False, True, False, False],
                         kernel_sizes=[3, 5, 7, 9],
                         d2_scales=[2, 2, 3, 4],
                         num_classes=10,  # CIFAR-10 classes
                         **kwargs)
    return model
