import re


def sam_to_monai_sam_state_dict(sam_sam_state_dict: dict) -> dict:
    sam_monai_state_dict = sam_sam_state_dict.copy()

    img_encoder_mlp_w_pattern = re.compile(r"image_encoder\.blocks\.(\d+)\.mlp\.lin(\d+)\.weight")
    img_encoder_mlp_b_pattern = re.compile(r"image_encoder\.blocks\.(\d+)\.mlp\.lin(\d+)\.bias")
    img_encoder_attn_out_proj_w_pattern = re.compile(r"image_encoder\.blocks\.(\d+)\.attn\.proj.weight")
    img_encoder_attn_out_proj_b_pattern = re.compile(r"image_encoder\.blocks\.(\d+)\.attn\.proj.bias")
    img_encoder_attn_rel_pos_pattern = re.compile(r"image_encoder\.blocks\.(\d+)\.attn.rel_pos_([a-zA-Z])")
    mask_decoder_mlp_w_pattern = re.compile(r"mask_decoder\.transformer\.layers\.(\d+)\.mlp\.lin(\d+)\.weight")
    mask_decoder_mlp_b_pattern = re.compile(r"mask_decoder\.transformer\.layers\.(\d+)\.mlp\.lin(\d+)\.bias")

    for k, v in sam_sam_state_dict.items():
        match = img_encoder_mlp_w_pattern.match(k)
        if match:
            block_nbr = int(match.group(1))
            layer_nbr = int(match.group(2))
            new_key = f"image_encoder.blocks.{block_nbr}.mlp.layers.{layer_nbr - 1}.weight"
            sam_monai_state_dict[new_key] = v
            del sam_monai_state_dict[k]

        match = img_encoder_mlp_b_pattern.match(k)
        if match:
            block_nbr = int(match.group(1))
            layer_nbr = int(match.group(2))
            new_key = f"image_encoder.blocks.{block_nbr}.mlp.layers.{layer_nbr - 1}.bias"
            sam_monai_state_dict[new_key] = v
            del sam_monai_state_dict[k]

        match = img_encoder_attn_out_proj_w_pattern.match(k)
        if match:
            block_nbr = int(match.group(1))
            new_key = f"image_encoder.blocks.{block_nbr}.attn.out_proj.weight"
            sam_monai_state_dict[new_key] = v
            del sam_monai_state_dict[k]

        match = img_encoder_attn_out_proj_b_pattern.match(k)
        if match:
            block_nbr = int(match.group(1))
            new_key = f"image_encoder.blocks.{block_nbr}.attn.out_proj.bias"
            sam_monai_state_dict[new_key] = v
            del sam_monai_state_dict[k]

        match = img_encoder_attn_rel_pos_pattern.match(k)
        if match:
            block_nbr = int(match.group(1))
            rel_pos_dim = match.group(2)
            rel_pos_mapping = {"h": 0, "w": 1}
            new_key = f"image_encoder.blocks.{block_nbr}.attn.rel_positional_embedding.rel_pos_arr.{rel_pos_mapping[rel_pos_dim]}"
            sam_monai_state_dict[new_key] = v
            del sam_monai_state_dict[k]

        match = mask_decoder_mlp_w_pattern.match(k)
        if match:
            first_layer = int(match.group(1))
            second_layer = int(match.group(2))
            new_key = f"mask_decoder.transformer.layers.{first_layer}.mlp.layers.{second_layer - 1}.weight"
            sam_monai_state_dict[new_key] = v
            del sam_monai_state_dict[k]

        match = mask_decoder_mlp_b_pattern.match(k)
        if match:
            first_layer = int(match.group(1))
            second_layer = int(match.group(2))
            new_key = f"mask_decoder.transformer.layers.{first_layer}.mlp.layers.{second_layer - 1}.bias"
            sam_monai_state_dict[new_key] = v
            del sam_monai_state_dict[k]

    return sam_monai_state_dict
