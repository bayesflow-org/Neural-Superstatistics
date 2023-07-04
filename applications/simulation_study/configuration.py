default_settings = {
    "lstm1_hidden_units": 512,
    "lstm2_hidden_units": 256,
    "lstm3_hidden_units": 128,
    "trainer": {
        "checkpoint_path": "../checkpoints/simulation_study",
        "max_to_keep": 1,
        "default_lr": 5e-4,
        "memory": False,
    },
    "local_amortizer_settings": {
        "num_coupling_layers": 8,
        "coupling_design": 'interleaved'
    },
    "global_amortizer_settings": {
        "num_coupling_layers": 6,
        "coupling_design": 'interleaved'
    },
}

default_priors = {
    "ddm_shape": (5.0, 4.0, 1.5),
    "ddm_scale": (1/3, 1/3, 1/5),
    "scale_alpha": 1,
    "scale_beta": 25,
    "variability_loc": (0.0, 0.0, 0.0),
    "variability_scale": (0.1, 0.1, 0.1)
}

default_lower_bounds = (0.0, 0.0, 0.0)
default_upper_bounds = (8.0, 6.0, 4.0)
default_points_of_jump = (100, 200, 300)
default_num_steps = {'default_num_steps': 400}
