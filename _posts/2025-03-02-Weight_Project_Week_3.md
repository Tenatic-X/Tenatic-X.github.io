---
layout: post
tags: Weight/Waist_prediction_project
todolist: Title, Overview, Projects and Tasks, Challenges and Solutions, Learnings and Insights, Next Steps, Reflections
---

# Week 3

## Overview/Tasks

Continuing the weight and waist project, using different ML models, tweaking hyperparameters, and using known ways to improve model performance.

## Challenges and solutions

Using `Lasso` and `ElasticNet` (the two suggested ML models based on the Scikit-learn cheatsheet), did not provide fruitful results. Most of the time, my custom `MAPE` evaluator shows a score hovering around 100. And if it's not around 100, the value is higher! I felt pretty disheartened by the result, as based on the rules I've created for `MAPE`, predicting a value of 0 when the real value is not 0, gives you an accuracy scare of 100. The hypothesis is that the model is mostly predicted around or at 0, as it was a safer value than to risk predicting a value. Testing through many hyperparameters, and also using optimizations to improve model performance, which did not help.
![dd6717bd-38db-423e-9e4c-acf3b716ed05](https://github.com/user-attachments/assets/b2aa6deb-5477-49e6-83da-3270d1e79a48)
And my hypothesis proves me correct.

## Learnings and Insights

* ElasticNet/Lasso are linear models: After talking with ChatGPT, it said that the models that I use, are really useful on data that has a linear pattern. But my one is non-linear, so it doesn't work well with this data.
* Values close to 0: ChatGPT has stated, that values hovering at or around 0, make it very difficult for models to learn any valueable patterns, as it's fed with 0 constantly.
* Using TensorFlow model layers: I spent some time working through a TensorFlow course, and thought this is a good chance to use what I've learnt and written on this project. I've seen how accurate the model can get, and it's ability to predict non-linear patterns. So that's something I will be working towards.

  ## Next steps

Continue the ML model building stage, using TensorFlow.
