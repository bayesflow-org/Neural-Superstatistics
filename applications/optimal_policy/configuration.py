default_settings = {
    "lstm1_hidden_units": 512,
    "lstm2_hidden_units": 128,
    "transformer_hidden_units": 128,
    "trainer": {
        "checkpoint_path": "../checkpoints/coal_mining",
        "max_to_keep": 1,
        "default_lr": 5e-4,
        "memory": False,
    },
}
