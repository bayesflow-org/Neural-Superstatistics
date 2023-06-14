default_settings = {
    "gru_hidden_units": 128,
    "lstm1_hidden_units": 128,
    "lstm2_hidden_units": 128,
    "hidden_units_local": 128,
    "hidden_units_global": 128,
    "trainer": {
        "checkpoint_path": "../checkpoints/coal_mining",
        "max_to_keep": 1,
        "default_lr": 1e-4,
        "memory": False,
    },
}
