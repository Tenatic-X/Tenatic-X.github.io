---
layout: post
tags: Weight/Waist_prediction_project
todolist: Title, Overview, Projects and Tasks, Challenges and Solutions, Learnings and Insights, Next Steps, Reflections
---

## Overview/Tasks

Continuing the weight and waist project, creating evaluation functions, and doing some ML model training on our data sets.

## Challenges and solutions

learnt my MAPE had issues - dividing 0 to get infinity. using y_true with value 0s.
The first challenge was I wanted to use a function other than the usual R^2, or MSE. The problem was the value difference of weight and waist are very miniscule and 99% decimal values. Using R^2, and MSA would be difficult to fully evaluate, as the numbers are already that small.

So I thought of create a percentage based evaluation function, which doesn't exist in sklearn, and had to create the function myself, which led to a few issues.

Here is the initial code:
```
def mean_absolute_percent_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs(y_true - y_pred) / y_true) * 100 
```

Biggest issue here was that there's 0 in my dataset, and y_true being 0, will cause errors as you cant divide by zero. So there were rules set in the function to circumnavigate the error. IF 0 is present in `true`, but there's a value in `prediction`, set the percentage difference to 100%. If both `true` and `prediction` is 0, set difference to 0%. Then do the rest as is with the mathematical operations.

Next challenge was ML model, and tweaking them. So far using Lasso, I've tried normalizing/scaling my data, and did some Grid Search CV, which hasn't provided any promising results, that even match the default ML model training that Lasso gave me. So more needs to be tested.

## Learnings and Insights

* Not everything is as simple as it looks.
  Despite how easy I thought my `Mean Average Percentage Error` evalution metric would be to create in my function as its just percentage difference finding, there's always little but very important rules hidden within it, that you'll have to figure out and problem solve, and it just so happens my dataset is very prone to these small unforseen errors.

* Don't always expect your model tweaking to go in your favour
  I get my hopes up, thinking that the tweaking I've done to my model will for sure contribute to better model accuracy. But it's always a work in progress, and you'll need to do a fair bit more playing around to figure out the best presets, hyperparameters, and ML model itself.

  ## Next steps

I'll be temporarily be studying/working on another ML computer science course, and stop my work on my first project. Biggest problem was I forgot to fucking save my project onto my portable SSD, and have an outdated version of the project with me overseas. So I'll have to wait till I'm back home to resume work lmao.

At least there's many more resources for me to study and upskill myself while away from home.
