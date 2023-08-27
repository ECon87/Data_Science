# A gentle introduction to Gradient Boosting

## Learning objectives of this chapter
- Origin of boosting from learning theory and AdaBoost.
- How gradient boosting works including the loss function, weak learners, and the additive model.
- How to improve performance over the base algorithm with various regularization schemes.


## Origin of Boosting
- Idea of Boosting is that a weak learner can be modified to become better.
- Weak hypothesis/learner is one whose performance is at least slightly better than random choice.
- Hypothesis boosting was the idea of filtering observations, leaving those observations that the weak learning can handle and focusing on developing new weak learners to handle the remaining difficult observations.


## AdaBoost the First Boosting Algorithm
- Adaptive Boosting (AdaBoost) was an early successful realization of boosting.
- The weak learnings in AdaBoost are decision trees with a single split, called decision stumps since they are very short trees.
- How does AdaBoost work: Weights observations by putting more weight on difficult to classify instances and less on those already handle well. New weak learners are added sequentially that focus on training on the more difficult patterns.

## Generalization of AdaBoost as Gradient Boosting
- AdaBoost and related algorithms were recast in Statistical Framework, called gradient boosting or gradient tree boosting.
- Reformed boosting as a numerical optimization problem where the objective is to minimize the loss of the model by adding weak learners using a gradient descent like procedure.
- This class of algorithms is described as a stage-wise additive model -- one new weak learner is added at a time and existing weak learners in the model are frozen and left unchanged.
- Generalization allowed arbitrary differentiable loss functions to be used (which allowed to expand applications to regression, multi-class classification, etc.)

