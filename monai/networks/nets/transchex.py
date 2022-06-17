# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import shutil
import tarfile
import tempfile
from typing import Sequence, Tuple, Union

import torch
from torch import nn

from monai.utils import optional_import

transformers = optional_import("transformers")
load_tf_weights_in_bert = optional_import("transformers", name="load_tf_weights_in_bert")
cached_path = optional_import("transformers.file_utils", name="cached_path")[0]
BertEmbeddings = optional_import("transformers.models.bert.modeling_bert", name="BertEmbeddings")[0]
BertLayer = optional_import("transformers.models.bert.modeling_bert", name="BertLayer")[0]

__all__ = ["BertPreTrainedModel", "BertAttention", "BertOutput", "BertMixedLayer", "Pooler", "MultiModal", "Transchex"]


class BertPreTrainedModel(nn.Module):
    """Module to load BERT pre-trained weights.
    Based on:
    LXMERT
    https://github.com/airsplay/lxmert
    BERT (pytorch-transformer)
    https://github.com/huggingface/transformers
    """

    def __init__(self, *inputs, **kwargs) -> None:
        super().__init__()

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(
        cls,
        num_language_layers,
        num_vision_layers,
        num_mixed_layers,
        bert_config,
        state_dict=None,
        cache_dir=None,
        from_tf=False,
        *inputs,
        **kwargs,
    ):
        archive_file = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz"
        resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            tempdir = tempfile.mkdtemp()
            with tarfile.open(resolved_archive_file, "r:gz") as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        model = cls(num_language_layers, num_vision_layers, num_mixed_layers, bert_config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, "pytorch_model.bin")
            state_dict = torch.load(weights_path, map_location="cpu" if not torch.cuda.is_available() else None)
        if tempdir:
            shutil.rmtree(tempdir)
        if from_tf:
            weights_path = os.path.join(serialization_dir, "model.ckpt")
            return load_tf_weights_in_bert(model, weights_path)
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        start_prefix = ""
        if not hasattr(model, "bert") and any(s.startswith("bert.") for s in state_dict.keys()):
            start_prefix = "bert."
        load(model, prefix=start_prefix)
        return model


class BertAttention(nn.Module):
    """BERT attention layer.
    Based on: BERT (pytorch-transformer)
    https://github.com/huggingface/transformers
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.dropout(nn.Softmax(dim=-1)(attention_scores))
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertOutput(nn.Module):
    """BERT output layer.
    Based on: BERT (pytorch-transformer)
    https://github.com/huggingface/transformers
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertMixedLayer(nn.Module):
    """BERT cross attention layer.
    Based on: BERT (pytorch-transformer)
    https://github.com/huggingface/transformers
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.att_x = BertAttention(config)
        self.output_x = BertOutput(config)
        self.att_y = BertAttention(config)
        self.output_y = BertOutput(config)

    def forward(self, x, y):
        output_x = self.att_x(x, y)
        output_y = self.att_y(y, x)
        return self.output_x(output_x, x), self.output_y(output_y, y)


class Pooler(nn.Module):
    """BERT pooler layer.
    Based on: BERT (pytorch-transformer)
    https://github.com/huggingface/transformers
    """

    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MultiModal(BertPreTrainedModel):
    """
    Multimodal Transformers From Pretrained BERT Weights"
    """

    def __init__(
        self, num_language_layers: int, num_vision_layers: int, num_mixed_layers: int, bert_config: dict
    ) -> None:
        """
        Args:
            num_language_layers: number of language transformer layers.
            num_vision_layers: number of vision transformer layers.
            bert_config: configuration for bert language transformer encoder.

        """
        super().__init__()
        self.config = type("obj", (object,), bert_config)
        self.embeddings = BertEmbeddings(self.config)
        self.language_encoder = nn.ModuleList([BertLayer(self.config) for _ in range(num_language_layers)])
        self.vision_encoder = nn.ModuleList([BertLayer(self.config) for _ in range(num_vision_layers)])
        self.mixed_encoder = nn.ModuleList([BertMixedLayer(self.config) for _ in range(num_mixed_layers)])
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, vision_feats=None, attention_mask=None):
        language_features = self.embeddings(input_ids, token_type_ids)
        for layer in self.vision_encoder:
            vision_feats = layer(vision_feats, None)[0]
        for layer in self.language_encoder:
            language_features = layer(language_features, attention_mask)[0]
        for layer in self.mixed_encoder:
            language_features, vision_feats = layer(language_features, vision_feats)
        return language_features, vision_feats


class Transchex(torch.nn.Module):
    """
    TransChex based on: "Hatamizadeh et al.,TransCheX: Self-Supervised Pretraining of Vision-Language
    Transformers for Chest X-ray Analysis"
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[int, Tuple[int, int]],
        num_classes: int,
        num_language_layers: int,
        num_vision_layers: int,
        num_mixed_layers: int,
        hidden_size: int = 768,
        drop_out: float = 0.0,
        attention_probs_dropout_prob: float = 0.1,
        gradient_checkpointing: bool = False,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        intermediate_size: int = 3072,
        layer_norm_eps: float = 1e-12,
        max_position_embeddings: int = 512,
        model_type: str = "bert",
        num_attention_heads: int = 12,
        num_hidden_layers: int = 12,
        pad_token_id: int = 0,
        position_embedding_type: str = "absolute",
        transformers_version: str = "4.10.2",
        type_vocab_size: int = 2,
        use_cache: bool = True,
        vocab_size: int = 30522,
        chunk_size_feed_forward: int = 0,
        is_decoder: bool = False,
        add_cross_attention: bool = False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            num_classes: number of classes if classification is used.
            num_language_layers: number of language transformer layers.
            num_vision_layers: number of vision transformer layers.
            num_mixed_layers: number of mixed transformer layers.
            drop_out: faction of the input units to drop.

        The other parameters are part of the `bert_config` to `MultiModal.from_pretrained`.

        Examples:

        .. code-block:: python

            # for 3-channel with image size of (224,224), patch size of (32,32), 3 classes, 2 language layers,
            # 2 vision layers, 2 mixed modality layers and dropout of 0.2 in the classification head
            net = Transchex(in_channels=3,
                                 img_size=(224, 224),
                                 num_classes=3,
                                 num_language_layers=2,
                                 num_vision_layers=2,
                                 num_mixed_layers=2,
                                 drop_out=0.2)

        """
        super().__init__()
        bert_config = {
            "attention_probs_dropout_prob": attention_probs_dropout_prob,
            "classifier_dropout": None,
            "gradient_checkpointing": gradient_checkpointing,
            "hidden_act": hidden_act,
            "hidden_dropout_prob": hidden_dropout_prob,
            "hidden_size": hidden_size,
            "initializer_range": initializer_range,
            "intermediate_size": intermediate_size,
            "layer_norm_eps": layer_norm_eps,
            "max_position_embeddings": max_position_embeddings,
            "model_type": model_type,
            "num_attention_heads": num_attention_heads,
            "num_hidden_layers": num_hidden_layers,
            "pad_token_id": pad_token_id,
            "position_embedding_type": position_embedding_type,
            "transformers_version": transformers_version,
            "type_vocab_size": type_vocab_size,
            "use_cache": use_cache,
            "vocab_size": vocab_size,
            "chunk_size_feed_forward": chunk_size_feed_forward,
            "is_decoder": is_decoder,
            "add_cross_attention": add_cross_attention,
        }
        if not (0 <= drop_out <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if (img_size[0] % patch_size[0] != 0) or (img_size[1] % patch_size[1] != 0):  # type: ignore
            raise ValueError("img_size should be divisible by patch_size.")

        self.multimodal = MultiModal.from_pretrained(
            num_language_layers=num_language_layers,
            num_vision_layers=num_vision_layers,
            num_mixed_layers=num_mixed_layers,
            bert_config=bert_config,
        )

        self.patch_size = patch_size
        self.num_patches = (img_size[0] // self.patch_size[0]) * (img_size[1] // self.patch_size[1])  # type: ignore
        self.vision_proj = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_size, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.norm_vision_pos = nn.LayerNorm(hidden_size)
        self.pos_embed_vis = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        self.pooler = Pooler(hidden_size=hidden_size)
        self.drop = torch.nn.Dropout(drop_out)
        self.cls_head = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, token_type_ids=None, vision_feats=None):
        attention_mask = torch.ones_like(input_ids).unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        attention_mask = (1.0 - attention_mask) * -10000.0
        vision_feats = self.vision_proj(vision_feats).flatten(2).transpose(1, 2)
        vision_feats = self.norm_vision_pos(vision_feats)
        vision_feats = vision_feats + self.pos_embed_vis
        hidden_state_lang, hidden_state_vis = self.multimodal(
            input_ids=input_ids, token_type_ids=token_type_ids, vision_feats=vision_feats, attention_mask=attention_mask
        )
        pooled_features = self.pooler(hidden_state_lang)
        logits = self.cls_head(self.drop(pooled_features))
        return logits
