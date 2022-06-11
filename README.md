## Bayesian Optimization

In complex engineering problems we often come across parameters that have to be tuned using several time-consuming and noisy evaluations. When the number of parameters is not small or some of the parameters are continuous, using large factorial designs (e.g., “grid search”) or global optimization techniques for optimization require more evaluations than is practically feasible. These types of problems show up in a diversity of applications, such as

Tuning Internet service parameters and selection of weights for recommender systems,
Hyperparameter optimization for machine learning,
Finding optimal set of gait parameters for locomotive control in robotics, and
Tuning design parameters and rule-of-thumb heuristics for hardware design.
Bayesian optimization (BO) allows us to tune parameters in relatively few iterations by building a smooth model from an initial set of parameterizations (referred to as the "surrogate model") in order to predict the outcomes for as yet unexplored parameterizations. BO is an adaptive approach where the observations from previous evaluations are used to decide what parameterizations to evaluate next. The same strategy can be used to predict the expected gain from all future evaluations and decide on early termination, if the expected benefit is smaller than what is worthwhile for the problem at hand.

As you iterate over and over, the algorithm balances its needs of exploration and exploitation taking into account what it knows about the target function. At each step a Gaussian Process is fitted to the known samples (points previously explored), and the posterior distribution, combined with a exploration strategy (such as UCB (Upper Confidence Bound), or EI (Expected Improvement)), are used to determine the next point that should be explored

Tools:
- https://ax.dev/docs/bayesopt.html
- https://botorch.org/v/0.2.3/tutorials/fit_model_with_torch_optimizer
- https://github.com/fmfn/BayesianOptimization
- https://www.borealisai.com/en/blog/tutorial-8-bayesian-optimization/

Good Reads: 
- https://distill.pub/2020/bayesian-optimization/
- https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote15.html
- https://distill.pub/2020/bayesian-optimization/

Exploitation vs Exploration:
- https://github.com/fmfn/BayesianOptimization/blob/master/examples/exploitation_vs_exploration.ipynb

Visualization:
- https://philipperemy.github.io/visualization/

Design of Experiments
- https://danmackinlay.name/notebook/design_of_experiments.html
- https://towardsdatascience.com/design-optimization-with-ax-in-python-957b1fec776f

Others:
- https://www.youtube.com/watch?v=BQ4kVn-Rt84&ab_channel=paretos
- https://towardsdatascience.com/understanding-gaussian-process-the-socratic-way-ba02369d804
- https://blog.dominodatalab.com/fitting-gaussian-process-models-python
- https://medium.com/panoramic/gaussian-processes-for-little-data-2501518964e4
- https://juanitorduz.github.io/gaussian_process_reg/
- https://towardsdatascience.com/quick-start-to-gaussian-process-regression-36d838810319
- https://towardsdatascience.com/bayesian-optimization-with-python-85c66df711ec
