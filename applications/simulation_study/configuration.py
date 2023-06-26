default_settings = {
    "lstm1_hidden_units": 512,
    "lstm2_hidden_units": 128,
    "transformer_template_dim": 128,
    "transformer_summary_dim": 32,
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
default_lower_bounds = (0.0, 0.0, 0.0)
default_upper_bounds = (8.0, 6.0, 4.0)
default_ddm_params_prior_loc = (0.0, 0.0, 0.0)
default_ddm_params_prior_scale = (2.5, 2.5, 1.0)
default_scale_prior_loc = (0.0, 0.0, 0.0)
default_scale_prior_scale = (0.1, 0.1, 0.1)
default_variability_prior_loc = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
default_variability_prior_scale = (2.5, 2.5, 1.0, 1.0, 1.0, 1.0)
default_points_of_jump = (100, 200, 300)

default_num_steps = {'default_num_steps': 400}
