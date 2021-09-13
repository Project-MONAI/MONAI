# Copyright 2020 - 2021 MONAI Consortium
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
from typing import Sequence, Union

import torch
from torch import nn

from monai.utils import optional_import

transformers = optional_import("transformers")
load_tf_weights_in_bert = optional_import("transformers", name="load_tf_weights_in_bert")
cached_path, _ = optional_import("transformers.file_utils", name="cached_path")
BertEmbeddings, _ = optional_import("transformers.models.bert.modeling_bert", name="BertEmbeddings")
BertLayer, _ = optional_import("transformers.models.bert.modeling_bert", name="BertLayer")


class BertPreTrainedModel(nn.Module):
    """Module to load BERT pre-trained weights.
    Based on:
    LXMERT
    https://github.com/airsplay/lxmert

    BERT (pytorch-transformer)
    https://github.com/huggingface/transformers
    """

    def __init__(self, *inputs, **kwargs) -> None:
        super(BertPreTrainedModel, self).__init__()

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, state_dict=None, cache_dir=None, from_tf=False, *inputs, **kwargs):
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
        model = cls(*inputs, **kwargs)
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

    def __init__(
        self,
        config,
    ) -> None:
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
        super(BertOutput, self).__init__()
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

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.att = BertAttention(config)
        self.output = BertOutput(config)

    def forward(self, x, y):
        output = self.att(x, y)
        return self.output(output, x)


class Pooler(nn.Module):
    """BERT pooler layer.
    Based on: BERT (pytorch-transformer)
    https://github.com/huggingface/transformers
    """

    def __init__(
        self,
        hidden_size,
    ) -> None:
        super(Pooler, self).__init__()
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
        self,
        num_language_layers: int = 2,
        num_vision_layers: int = 2,
        num_mixed_layers: int = 2,
    ) -> None:
        """
        Args:
            num_language_layers: number of language transformer layers.
            num_vision_layers: number of vision transformer layers.
            num_mixed_layers: number of mixed transformer layers.
        """
        super().__init__()
        bert_config = type(
            "obj",
            (object,),
            {
                "attention_probs_dropout_prob": 0.1,
                "classifier_dropout": None,
                "gradient_checkpointing": False,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "hidden_size": 768,
                "initializer_range": 0.02,
                "intermediate_size": 3072,
                "layer_norm_eps": 1e-12,
                "max_position_embeddings": 512,
                "model_type": "bert",
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
                "pad_token_id": 0,
                "position_embedding_type": "absolute",
                "transformers_version": "4.10.2",
                "type_vocab_size": 2,
                "use_cache": True,
                "vocab_size": 30522,
                "chunk_size_feed_forward": 0,
                "is_decoder": False,
                "add_cross_attention": False,
            },
        )

        self.config = bert_config
        self.embeddings = BertEmbeddings(bert_config)
        self.language_encoder = nn.ModuleList([BertLayer(bert_config) for _ in range(num_language_layers)])
        self.vision_encoder = nn.ModuleList([BertLayer(bert_config) for _ in range(num_vision_layers)])
        self.mixed_encoder = nn.ModuleList([BertMixedLayer(bert_config) for _ in range(num_mixed_layers)])
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, vision_feats=None, attention_mask=None):
        language_features = self.embeddings(input_ids, token_type_ids)
        for layer in self.vision_encoder:
            hidden_state_vision = layer(vision_feats, None)[0]
        for layer in self.language_encoder:
            hidden_state_language = layer(language_features, attention_mask)[0]
        for layer in self.mixed_encoder:
            hidden_state_mixed = layer(hidden_state_language, hidden_state_vision)
        return hidden_state_mixed


class VLTransformers(torch.nn.Module):
    """
    Vision Language Multimodal Transformers"
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        num_classes: int = 2,
        num_language_layers: int = 2,
        num_vision_layers: int = 2,
        num_mixed_layers: int = 2,
        drop_out: float = 0.1,
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

        Examples::

            # for 3-channel with image size of (224,224), patch size of (32,32), 3 classes, 2 language layers,
            2 vision layers and 2 mixed modality layers
            >>> net = VLTransformers(in_channels=3, img_size=(224, 224), num_classes=3, num_language_layers=2,
            num_vision_layers=2, num_mixed_layers=2)

        """
        super(VLTransformers, self).__init__()

        if not (0 <= drop_out <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if (img_size[0] % patch_size[0] != 0) or (img_size[1] % patch_size[1] != 0):
            raise ValueError("img_size should be divisible by patch_size.")

        self.multimodal = MultiModal(
            num_language_layers=num_language_layers,
            num_vision_layers=num_vision_layers,
            num_mixed_layers=num_mixed_layers,
        ).from_pretrained()
        self.embed_dim = 768
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // self.patch_size[0]) * (img_size[1] // self.patch_size[1])
        self.vision_proj = nn.Conv2d(in_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm_vision_pos = nn.LayerNorm(self.embed_dim)
        self.pos_embed_vis = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        self.pooler = Pooler(hidden_size=self.embed_dim)
        self.drop = torch.nn.Dropout(drop_out)
        self.cls_head = torch.nn.Linear(self.embed_dim, num_classes)

    def forward(self, input_ids, token_type_ids=None, vision_feats=None):
        attention_mask = torch.ones_like(input_ids).unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        attention_mask = (1.0 - attention_mask) * -10000.0
        vision_feats = self.vision_proj(vision_feats).flatten(2).transpose(1, 2)
        vision_feats = self.norm_vision_pos(vision_feats)
        vision_feats = vision_feats + self.pos_embed_vis
        hidden_state_mixed = self.multimodal(
            input_ids=input_ids, token_type_ids=token_type_ids, vision_feats=vision_feats, attention_mask=attention_mask
        )
        pooled_features = self.pooler(hidden_state_mixed)
        logits = self.cls_head(self.drop(pooled_features))
        return logits
