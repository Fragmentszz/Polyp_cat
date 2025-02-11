reins_config=dict(
    type="LoRAReins",
    token_length=128,
    embed_dims=1280,
    num_layers=4,
    patch_size=16,
    link_token_to_query=True,
    lora_dim=16,
    zero_mlp_delta_f=False,  # v2
)