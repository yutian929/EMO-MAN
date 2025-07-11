import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = ["init_modules", "zero_last_gamma"]


def init_modules(model: nn.Module | list[nn.Module], init_type="trunc_normal") -> None:
    _DEFAULT_INIT_PARAM = {"trunc_normal": 0.02}

    if isinstance(model, list):
        for sub_module in model:
            init_modules(sub_module, init_type)
    else:
        init_params = init_type.split("@")
        init_params = float(init_params[1]) if len(init_params) > 1 else None

        if init_type.startswith("trunc_normal"):
            init_func = lambda param: nn.init.trunc_normal_(
                param,
                std=(
                    _DEFAULT_INIT_PARAM["trunc_normal"]
                    if init_params is None
                    else init_params
                ),
            )
        else:
            raise NotImplementedError

        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                init_func(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                init_func(m.weight)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            else:
                weight = getattr(m, "weight", None)
                bias = getattr(m, "bias", None)
                if isinstance(weight, torch.nn.Parameter):
                    init_func(weight)
                if isinstance(bias, torch.nn.Parameter):
                    bias.data.zero_()


def zero_last_gamma(model: nn.Module, init_val=0) -> None:
    import efficientvit.models.nn.ops as ops

    for m in model.modules():
        if isinstance(m, ops.ResidualBlock) and isinstance(
            m.shortcut, ops.IdentityLayer
        ):
            if isinstance(m.main, (ops.DSConv, ops.MBConv, ops.FusedMBConv)):
                parent_module = m.main.point_conv
            elif isinstance(m.main, ops.ResBlock):
                parent_module = m.main.conv2
            elif isinstance(m.main, ops.ConvLayer):
                parent_module = m.main
            elif isinstance(m.main, (ops.LiteMLA)):
                parent_module = m.main.proj
            else:
                parent_module = None
            if parent_module is not None:
                norm = getattr(parent_module, "norm", None)
                if norm is not None:
                    nn.init.constant_(norm.weight, init_val)
