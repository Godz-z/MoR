# coding=utf-8
import math
import re
import warnings
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils import (
    COMMON_LAYERS_PATTERN,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftConfig,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)

if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class LoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`LoraModel`].

    Args:
        r (`int`): Lora attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`int`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the LoRA transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the LoRA
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
                    "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
                    "For example, in Sequence Classification or Token Classification tasks, "
                    "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the Lora layers."},
    )
    layers_to_transform: Optional[Union[List, int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.LORA


class LoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig
        >>> from peft import LoraModel, LoraConfig

        >>> config = LoraConfig(
        ...     peft_type="LORA",
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=["q", "v"],
        ...     lora_dropout=0.01,
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> lora_model = LoraModel(config, model)
        ```

        ```py
        >>> import transformers
        >>> from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training

        >>> target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        >>> config = LoraConfig(
        ...     r=4, lora_alpha=16, target_modules=target_modules, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
        ... )

        >>> model = transformers.GPTJForCausalLM.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     pad_token_id=tokenizer.eos_token_id,
        ...     use_cache=False,
        ...     device_map={"": rank},
        ...     torch_dtype=torch.float16,
        ...     load_in_8bit=True,
        ... )
        >>> model = prepare_model_for_int8_training(model)
        >>> lora_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, model, config, adapter_name, number_experts, top_k):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.number_experts = number_experts
        self.top_k = top_k
        self.add_adapter(adapter_name, self.number_experts, self.top_k, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, number_experts, top_k, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_lora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "LoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        # mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)  ##modified
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        print("TRAINING MOLA")

        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    # 检查是否提供了适配器配置config，如果提供了，则准备并设置Lora配置。
    # 调用_find_and_replace方法，该方法的作用通常是查找模型中需要替换的模块，并将其替换为Lora模块。
    # 检查是否有多个适配器配置，如果有且bias不为none，则抛出异常，因为该模型仅支持一个带偏置的适配器。
    # 调用mark_only_lora_as_trainable方法，标记Lora层为可训练。这里的注释modified表明代码进行了某些修改。
    # 打印信息"TRAINING MOLA"，表示开始训练MOLA（可能是Mixture
    # of Lora Adapter）的过程。
    # 如果适配器配置的inference_mode为真，则调用_freeze_adapter方法，将适配器冻结。
    def _check_quantization_dependency(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit) and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )

    # _check_quantization_dependency：此函数检查是否已加载模型的4 - bit或8 - bit量化版本，并且是否安装了bitsandbytes包。若没有安装bitsandbytes，则会抛出一个ImportError异常，提醒用户安装该包。

    def _check_target_module_exists(self, lora_config, key):
        if isinstance(lora_config.target_modules, str):
            target_module_found = re.fullmatch(lora_config.target_modules, key)
        else:
            target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            is_using_layer_indexes = getattr(lora_config, "layers_to_transform", None) is not None
            layer_indexing_pattern = getattr(lora_config, "layers_pattern", None)

            if is_using_layer_indexes and target_module_found:
                layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

                for pattern in layers_pattern:
                    layer_index = re.match(f".*.{pattern}\.(\d+)\.*", key)
                    if layer_index is not None:
                        layer_index = int(layer_index.group(1))
                        if isinstance(lora_config.layers_to_transform, int):
                            target_module_found = layer_index == lora_config.layers_to_transform
                        else:
                            target_module_found = layer_index in lora_config.layers_to_transform

                        break
                    else:
                        target_module_found = False
        return target_module_found

    # 这段代码确保了在使用量化模型时安装了必要的依赖项，并检查给定的key是否符合LoRA配置中的目标模块。
    def _create_new_module(self, lora_config, adapter_name, target, layer_index, number_experts, top_k):  ### modified2
        bias = hasattr(target, "bias") and target.bias is not None
        # 这行代码检查目标层是否有偏置项，并将结果存储在bias变量中。
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,

        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        # 检查模型是否加载为4 - bit或8 - bit量化版本。
        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(
                adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
            )
        # 如果目标层是8 - bit量化的Linear8bitLt层，则复制kwargs并添加特定于8 - bit量化层的参数，然后创建Linear8bitLt模块。
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
        # 如果目标层是Embedding层，则创建Embedding模块。
        elif isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        # 如果目标层是Conv2d层，则创建Conv2d模块。
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = Linear_MoE(adapter_name, in_features, out_features, bias=bias, layer_index=layer_index,
                                    number_experts=number_experts, top_k=top_k, **kwargs)  ### modified2 ##newmodified
        # 如果目标层是Linear或Conv1D层，则创建Linear_MoE模块。如果目标层是Linear且fan_in_fan_out设置为True，则警告并设置为False。
        # 如果目标层是Conv1D且fan_in_fan_out设置为False，则警告并设置为True。如果目标层不是支持的类型，则抛出ValueError异常。
        return new_module

    # 这段代码定义了一个函数
    # _create_new_module，其目的是根据LoRA配置创建一个新的模块。它根据输入的目标层的类型及其量化状态（4 - bit或8 - bit），创建适当类型的新模块。
    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        # 从self.peft_config中获取指定适配器的LoRA配置。
        self._check_quantization_dependency()
        # 调用_check_quantization_dependency函数以确保必要的量化依赖项已安装。
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
        # 初始化一个标志is_target_modules_in_base_model为False，
        # 并创建模型中所有模块键的列表key_list。
        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)
            layer_index = int(key.split(".")[2])  ### modified2
            # 将标志is_target_modules_in_base_model设置为True，表示找到了目标模块。
            # 调用_get_submodules函数获取父模块、目标模块和目标模块名称。
            # 从键中提取层索引。
            if isinstance(target, LoraLayer) and isinstance(target, torch.nn.Conv2d):
                target.update_layer_conv2d(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
                # 如果目标模块是LoraLayer且是Conv2d类型，则调用update_layer_conv2d方法更新该模块。
            elif isinstance(target, LoraLayer):
                target.update_layer(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
                # 如果目标模块是LoraLayer，则调用update_layer方法更新该模块。
            else:
                new_module = self._create_new_module(lora_config, adapter_name, target, layer_index,
                                                     self.number_experts, self.top_k)  ### modified2
                self._replace_module(parent, target_name, new_module, target, layer_index)  ### modified2
        # 如果目标模块不是LoraLayer，则调用_create_new_module方法创建一个新模块，并调用_replace_module方法替换目标模块。
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    # 这段代码定义了一个函数_find_and_replace，其目的是在模型中找到目标模块并替换成新的模块。
    # 它首先检查量化依赖项，然后在模型中找到目标模块，根据模块的类型更新或替换它们。
    def _replace_module(self, parent_module, child_name, new_module, old_module, layer_index):  ### modified2
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        ##newmodified
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)
            if "router" in name:
                module.to(old_module.weight.device)
            if "ranknum" in name:
                module.to(old_module.weight.device)

    # 这段代码定义了一个函数_replace_module，用于在模型中替换指定的子模块，并确保新模块继承旧模块的权重、偏置和其他状态。该函数还会将新模块及其子模块分派到适当的设备上。
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    # 这个方法的作用是当访问的属性在当前类中不存在时，将该访问请求转发给被包装的模型（self.model）。
    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    # 这个方法将PEFT配置转换为字典格式，方便进一步处理或序列化。
    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.merge()

    def unmerge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.unmerge()

    @staticmethod
    def _prepare_lora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        return peft_config

    # 这个方法的主要功能是确保peft_config中的target_modules
    # 已正确设置。如果target_modules没有指定，则根据模型类型从一个预定义的映射中获取相应的默认目标模块。
    def merge_and_unload(self):
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        if getattr(self.config, "model_type", None) == "gpt2":
            raise ValueError("GPT2 models are not supported for merging LORA layers")
        # 如果模型类型是GPT2，则抛出异常，因为不支持合并LoRA层。
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")
        # 如果模型加载为8 - bit或4 - bit量化模式，则抛出异常，因为在这种模式下不支持合并LoRA层。
        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        # 获取所有不包含“lora”字符串的模块的键列表。
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            # 遍历键列表，尝试获取父模块、目标模块和目标模块名称。如果发生属性错误，继续下一个键。
            if isinstance(target, LoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                else:
                    bias = target.bias is not None
                    new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                target.merge()
                self._replace_module(parent, target_name, new_module, target)
            # 如果目标模块是LoraLayer：检查其类型，分别处理nn.Embedding、nn.Conv2d和其他（通常是nn.Linear）类型。
            # 创建相应的新模块。调用target.merge()方法合并LoRA层。使用_replace_module方法将新模块替换旧模块。
            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])
        # 如果目标模块是ModulesToSaveWrapper类型，则将其设置为相应父模块的属性，确保任何额外的可训练模块都被保存。
        return self.model

    def add_weighted_adapter(self, adapters, weights, adapter_name):
        if len({self.peft_config[adapter].r for adapter in adapters}) != 1:
            raise ValueError("All adapters must have the same r value")
        # 代码开始时，使用集合理解和一个条件判断来确保输入的所有适配器adapters具有相同的r值（一个与LoRA相关的配置参数）。如果r值不相同，则抛出一个ValueError。
        self.peft_config[adapter_name] = replace(
            self.peft_config[adapters[0]], lora_alpha=self.peft_config[adapters[0]].r
        )
        # 使用replace函数复制adapters[0]（第一个适配器）的配置，并将新适配器的名称adapter_name设置到模型的peft_config中，同时调整lora_alpha值为第一个适配器的r值。
        self._find_and_replace(adapter_name)
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        _freeze_adapter(self.model, adapter_name)
        # 调用_find_and_replace(adapter_name)方法来处理模型中的相关结构，可能涉及到名称或参数的替换。
        # 调用mark_only_lora_as_trainable函数设置新适配器的偏置为可训练。
        # 调用_freeze_adapter函数冻结新适配器以外的部分。
        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                if adapter_name in target.lora_A:
                    target.lora_A[adapter_name].weight.data = target.lora_A[adapter_name].weight.data * 0.0
                    target.lora_B[adapter_name].weight.data = target.lora_B[adapter_name].weight.data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.lora_A:
                            continue
                        target.lora_A[adapter_name].weight.data += (
                                target.lora_A[adapter].weight.data * weight * target.scaling[adapter]
                        )
                        target.lora_B[adapter_name].weight.data += target.lora_B[adapter].weight.data * weight

                elif adapter_name in target.lora_embedding_A:
                    target.lora_embedding_A[adapter_name].data = target.lora_embedding_A[adapter_name].data * 0.0
                    target.lora_embedding_B[adapter_name].data = target.lora_embedding_B[adapter_name].data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.lora_embedding_A:
                            continue
                        target.lora_embedding_A[adapter_name].data += (
                                target.lora_embedding_A[adapter].data * weight * target.scaling[adapter]
                        )
                        target.lora_embedding_B[adapter_name].data += target.lora_embedding_B[adapter].data * weight


# 遍历模型的所有子模块，找出不含lora的模块名称，并检查是否是LoraLayer类型。
# 如果是，并且新适配器已经存在于lora_A或lora_embedding_A中，将相应的权重初始化为0。
# 对于每个原始适配器，如果它存在于目标模块中，按照给定的权重weights和可能的缩放因子scaling累加权重到新适配器中。这包括两部分：矩阵A和矩阵B的权重。
# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
##newmodified
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n and 'router' not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


# 函数接受一个模型（model）和一个字符串（bias），用于指定哪些类型的偏置（bias）参数应该是可训练的。
##modified
def mark_only_lora_B_as_trainable(model: nn.Module, bias: str = "none") -> None:
    print('*******************', 'ONLY TRAIN EXPERTS WITHOUT ROUTER', '*******************')
    # 开始时打印一行消息，提示正在对除'router'外的专家进行训练设置。
    # 与原始函数相比，修改的函数中去掉了对'router'的特殊处理，即不再检查参数名称中是否包含'router'。
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class LoraLayer:
    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    # 这些字典分别用于存储不同适配器的相关参数
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, layer_index):  ### modified2
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if isinstance(r, list):  ### modified2
            self.lora_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r[layer_index], bias=False)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: nn.Linear(r[layer_index], self.out_features, bias=False)}))
            self.scaling[adapter_name] = lora_alpha / r[layer_index]
        else:
            if r > 0:
                self.lora_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)}))
                self.lora_B.update(nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)}))
                self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    # 根据提供的r值更新self.r, self.lora_alpha。
    # 根据lora_dropout创建对应的Dropout层，如果为0则使用恒等变换。
    # 如果r是列表，则创建对应layer_index层的A和B矩阵，并更新缩放值。如果r是一个整数，则创建单一秩的A和B矩阵。
    # 如果init_lora_weights为真，则重置适配器的LoRA参数。
    # 将模型转移到当前权重所在的设备。
    def update_layer_conv2d(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            kernel_size = self.kwargs["kernel_size"]
            stride = self.kwargs["stride"]
            padding = self.kwargs["padding"]
            self.lora_A.update(
                nn.ModuleDict({adapter_name: nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)})
            )
            self.lora_B.update(
                nn.ModuleDict({adapter_name: nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)})
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    # 这个方法用于为卷积层添加或更新LoRA适配器。
    def update_layer_embedding(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_embedding_A.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((r, self.in_features)))})
            )
            self.lora_embedding_B.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((self.out_features, r)))})
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    # 这个方法用于为嵌入层添加或更新LoRA适配器。
    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])


# 这段代码定义了一个名为 reset_lora_parameters 的方法，用于重置LoRA适配器中的权重参数。这个方法主要处理两种类型的参数：线性变换（lora_A 和 lora_B）和嵌入参数（lora_embedding_A 和 lora_embedding_B）。
##newmodified
class LoraMoE_Layer:
    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # self.router = nn.ModuleDict({})
        # -------------------------------

        self.router_A = nn.ModuleDict({})
        self.router_1 = nn.ModuleDict({})
        self.router_2 = nn.ModuleDict({})
        # -------------------------------
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    # 这个构造器初始化了多个重要的属性
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, layer_index):  ### modified2
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # 设置适配器的秩和缩放因子。根据lora_dropout值决定使用Dropout或Identity（无操作层）。
        # Actual trainable parameters
        if isinstance(r, list):  ### modified2
            ##newmodified
            # print("LINEAR", self.number_experts)
            # self.router.update(nn.ModuleDict({'router': nn.Linear(self.in_features, self.number_experts, bias=False)}))
            # -------------------------
            self.router_1.update(
                nn.ModuleDict({'router': nn.Linear(self.in_features, self.number_experts, bias=False)}))
            self.router_2.update(
                nn.ModuleDict({'router': nn.Linear(self.in_features, self.number_experts, bias=False)}))
            self.router_A.update(
                nn.ModuleDict({'router': nn.Linear(self.in_features, 2, bias=True)}))
            # -------------------------
            for i in range(self.number_experts):
                adapter_name_moe = adapter_name + '_' + str(i)
                self.lora_A.update(
                    nn.ModuleDict({adapter_name_moe: nn.Linear(self.in_features, r[layer_index], bias=False)}))
                self.lora_B.update(
                    nn.ModuleDict({adapter_name_moe: nn.Linear(r[layer_index], self.out_features, bias=False)}))
            self.scaling[adapter_name] = lora_alpha / r[layer_index]
            # 如果r是列表（针对MoE架构），则更新router参数，这里假设已经有一个number_experts属性定义了专家的数量。
            # 对每个专家，创建独立的适配器（lora_A和lora_B），并将它们加入对应的模块字典中。这些适配器的命名遵循格式adapter_name_i，
            # 其中i是专家的索引。更新缩放值为lora_alpha / r[layer_index]。
        else:
            if r > 0:
                # self.lora_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)}))
                # self.lora_B.update(nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)}))
                # -------------------------
                self.router_1.update(
                    nn.ModuleDict({adapter_name: nn.Linear(self.in_features, self.number_experts, bias=False)}))
                self.router_2.update(
                    nn.ModuleDict({adapter_name: nn.Linear(self.in_features, self.number_experts, bias=False)}))
                self.router_A.update(
                    nn.ModuleDict({adapter_name: nn.Linear(self.in_features, 2, bias=False)}))
                self.lora_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)}))
                self.lora_B.update(nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)}))
                self.scaling[adapter_name] = lora_alpha / r
                # -------------------------
        # 如果 r 是单一整数且大于0，直接为这一适配器创建线性变换层 lora_A 和 lora_B。
        ##newmodified
        if init_lora_weights:
            if isinstance(r, list):
                self.reset_lora_parameters_MoE(adapter_name)
            else:
                self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    # 根据 r 的类型（列表或单一值）决定调用 reset_lora_parameters_MoE 或 reset_lora_parameters 方法来初始化权重。
    # 这种设计允许不同类型的初始化逻辑应用于不同的架构（MoE 或单一适配器）。
    def update_layer_conv2d(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            kernel_size = self.kwargs["kernel_size"]
            stride = self.kwargs["stride"]
            padding = self.kwargs["padding"]
            self.lora_A.update(
                nn.ModuleDict({adapter_name: nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)})
            )
            self.lora_B.update(
                nn.ModuleDict({adapter_name: nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)})
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_embedding(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_embedding_A.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((r, self.in_features)))})
            )
            self.lora_embedding_B.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((self.out_features, r)))})
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    # 这两个方法 update_layer_conv2d 和 update_layer_embedding 是类中的成员函数，用于更新或添加LoRA适配器配置，适用于不同类型的层（卷积层和嵌入层）。
    ##newmodified
    def reset_lora_parameters_MoE(self, adapter_name):
        # nn.init.kaiming_uniform_(self.router['router'].weight, a=math.sqrt(5))
        # -------------------------
        nn.init.kaiming_uniform_(self.router_1['router'].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.router_2['router'].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.router_A['router'].weight, a=math.sqrt(5))
        # -------------------------
        for i in range(self.number_experts):
            adapter_name_moe = adapter_name + '_' + str(i)
            if adapter_name_moe in self.lora_A.keys():
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A[adapter_name_moe].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[adapter_name_moe].weight)
            if adapter_name_moe in self.lora_embedding_A.keys():
                # initialize a the same way as the default for nn.linear and b to zero
                nn.init.zeros_(self.lora_embedding_A[adapter_name_moe])
                nn.init.normal_(self.lora_embedding_B[adapter_name_moe])

    #    参数设置：存储或更新适配器名称、秩（r）、缩放因子（lora_alpha）、dropout率（lora_dropout）。
    #    Dropout层配置：根据dropout率配置相应的dropout层；如果dropout为0，则使用Identity层。
    #    卷积层参数的配置：
    #    只有当秩 r 大于0时，才会创建卷积层。
    #    从kwargs读取kernel_size、stride和padding参数，设置卷积层参数。
    #    lora_A 是将输入特征转换到中间特征的卷积层，而 lora_B 是将中间特征映射回输出特征的卷积层，其中 lora_B 的卷积核大小固定为 (1, 1)，以维持特征图的空间尺寸。
    #    缩放比例的更新：更新适配器的缩放值为 lora_alpha / r。
    #    权重初始化和设备配置：如果指定初始化权重，则调用权重初始化函数，并确保所有操作在适当的计算设备上执行。
    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])


# 嵌入层参数的配置：只有当秩 r 大于0时，才会创建新的嵌入参数。
# 创建两个新的嵌入参数，lora_embedding_A 和 lora_embedding_B，分别用于嵌入向量的低秩调整和变换。
# 这些新参数用self.weight.new_zeros初始化为零，以保证与模型其他部分的兼容性。
##newmodified
class Linear_MoE(nn.Linear, LoraMoE_Layer):

    # 这个代码定义了一个名为 Linear_MoE 的类，它继承自 PyTorch 的 nn.Linear 和自定义的 LoraMoE_Layer。
    # 这个类融合了线性层、LoRA (Low-Rank Adaptation) 技术，以及 MoE (Mixture of Experts) 架构的特点。以下是对这个类的详细分析：
    # Lora implemented in a dense layer
    def __init__(
            self,
            adapter_name: str,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            layer_index: int = 0,  ## modified2
            number_experts: list = [8] * 32,
            top_k: list = [2] * 32,
            **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        ##newmodified
        LoraMoE_Layer.__init__(self, in_features=in_features, out_features=out_features)
        # 基础初始化：
        # 首先初始化基类 nn.Linear，设置输入和输出特征数量。
        # 接着初始化 LoraMoE_Layer，这使得类具有处理LoRA适配器和MoE特性的能力。
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        # 初始状态下，原始权重矩阵的梯度更新被禁用，以保持预训练权重不变。
        # 如果fan_in_fan_out为真，则转置权重矩阵。
        self.number_experts = number_experts[layer_index]
        self.top_k = top_k[layer_index]
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, layer_index)  ##modified2
        # 调用update_layer方法初始化或更新LoRA适配器和其参数。
        # 激活当前适配器adapter_name。
        self.active_adapter = adapter_name
        self.softmax = nn.Softmax(dim=-1)  ##modified
        self.sigmoid = nn.Sigmoid()
        # 配置了Softmax和Sigmoid函数，这可能用于后续处理输出或在MoE逻辑中选择专家。

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                    transpose(
                        self.lora_B[self.active_adapter].weight @ self.lora_A[self.active_adapter].weight,
                        self.fan_in_fan_out,
                    )
                    * self.scaling[self.active_adapter]
            )
            self.merged = True

    # 该方法用于将LoRA适配器计算的修改合并到原始权重中。
    # 如果适配器激活且未合并，则将lora_B和lora_A计算的结果（需要考虑是否转置）乘以缩放因子后加到原始权重上。
    # 通过merged标志位控制权重是否已合并。
    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                    transpose(
                        self.lora_B[self.active_adapter].weight @ self.lora_A[self.active_adapter].weight,
                        self.fan_in_fan_out,
                    )
                    * self.scaling[self.active_adapter]
            )
            self.merged = False


    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        # 基础的线性变换
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        batch_size, sequence_length, hidden_dim = x.shape
        x = x.view(-1, hidden_dim)
        # 从两个独立的路由器计算输出
        router_logits_1 = self.router_1['router'](x)
        router_logits_2 = self.router_2['router'](x)
        router_logits_A = self.router_A['router'](x)

        # 对 router_logits_C 进行 softmax，得到两个分数 x 和 y
        softmax_logits_A = F.softmax(router_logits_A, dim=1)  # dim=1 表示按最后一个维度进行 softmax
        total_0 = softmax_logits_A[:, 0].sum()
        total_1 = softmax_logits_A[:, 1].sum()
        total_sum = total_0 + total_1
        normalized_total_0 = total_0 / total_sum
        # 计算最终的 router_logits，仅使用 top-2 的归一化权重
        router_logits = normalized_total_0 * router_logits_1 + (1 - normalized_total_0) * router_logits_2
        # print("router_logits:", router_logits.shape)
        routing_weights_before = F.softmax(router_logits, dim=1)
        # print("routing_weights_before:", routing_weights_before.shape)
        routing_weights, selected_experts = torch.topk(routing_weights_before, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.number_experts).permute(2, 1, 0)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, self.out_features), dtype=x.dtype, device=x.device
        )
        for expert_idx in range(self.number_experts):
            adapter_name_moe = self.active_adapter + '_' + str(expert_idx)
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            #print(x[None, top_x_list].shape)
            # print(f"x shape: {x.shape}")
            # print(f"top_x_list length: {len(top_x_list)}")
            current_state = x[None, top_x_list].reshape(-1, hidden_dim)
            expert_output = (
                    self.lora_B[adapter_name_moe](
                        self.lora_A[adapter_name_moe](self.lora_dropout[self.active_adapter](current_state))
                    )
                    * self.scaling[self.active_adapter]
            )
            current_hidden_states = expert_output * routing_weights[top_x_list, idx_list, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, self.out_features)
        result += final_hidden_states
        result = result.to(previous_dtype)

        return result, router_logits



