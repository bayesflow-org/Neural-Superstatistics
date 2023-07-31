# Neural Superstatistics

This repository contains the code for running the experiments and reproducing all results reported in our paper [Neural Superstatistics for Bayesian Estimation of Dynamic Cognitive Models](https://arxiv.org/abs/2211.13165). We propose to augment mechanistic cognitive models with a temporal dimension and estimate the resulting dynamics from a superstatistics perspective.

The details of the method are described in our paper:

Schumacher, L., Bürkner, P. C., Voss, A., Köthe, U., & Radev, S. T. (2023). 
Neural Superstatistics for Bayesian Estimation of Dynamic Cognitive Models
<em>arXiv preprint arXiv:2211.13165</em>, available for free at: https://arxiv.org/abs/2211.13165.

The code depends on the [BayesFlow](https://github.com/stefanradev93/BayesFlow) library, which implements the neural network architectures and training utilities.

## Cite

```bibtex
@article{schumacher2022neural,
      title={Neural Superstatistics: A Bayesian Method for Estimating Dynamic Models of Cognition}, 
      author={Lukas Schumacher and Paul-Christian Bürkner and Andreas Voss and Ullrich Köthe and Stefan T. Radev},
      year={2022},
      journal={arXiv preprint arXiv:2211.13165}
}
```

## [applications](applications)

All applications are structured as self-contained Jupyter notebooks, which are detailed below.

### Benchmark studies:

- [Bayesloop benchmark](applications/coal_mining/notebooks/bayesloop_benchmark.ipynb): Comparison of our neural estimation method with the Bayesloop method, which is based on grid approximation.
- [Stan benchmark](applications/stan_comparison/notebooks/stan_benchmark.ipynb): Comparison of our neural estimation method with the HMC-based estimaton method Stan.

### Simulation studies:

- [Simulation study](applications/simulation_study/notebook/simulation_study_experiment.ipynb): Assessment of the parameter recovery performance of a non-stationary DDM in four different simulation scenarios.

### Human data applications:

- [Optimal policy](applications/optimal_policy/notebook/optimal_policy_experiment.ipynb): Fitting a non-stationary DDM to data from a standard random-dot motion task.
- [Lexical decision](applications/lexical_decision/notebooks): Fitting a non-stationary DDM to data from a lexical decision task.

## Parameter recovery

The following animation shows the parameter recovery performance of the a Gaussian Process DDM over 3200 time steps.

![](param_recovery_animation.gif)

## Support

This work was supported by the Deutsche Forschungsgemein-schaft (DFG, German Research Foundation; grant number GRK 2277 ”Statistical Modeling in Psychology”), by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy – EXC-2075 - 390740016 (the Stuttgart Cluster of Excellence SimTech), and by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy – EXC-2181 - 390900948 (the Heidelberg Cluster of Excellence STRUCTURES).

## License

MIT







