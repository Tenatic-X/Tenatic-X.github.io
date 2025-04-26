---
layout: post
tags: Weight/Waist_prediction_project
todolist: Title, Overview, Projects and Tasks, Challenges and Solutions, Learnings and Insights, Next Steps, Reflections
---

# Week 8 & 9

## Relevant Work
[Weight and waist workbook](https://tenatic-x.github.io/_projects/weight%20and%20waist%20workbook.html)

[Weight and waist pdf report](https://tenatic-x.github.io/_projects/workbook%20report.pdf)

## Overview/Tasks

Continuing the weight and waist project, now finalizing models, and also tweaking whatever is needed. In addition, create a pdf report of the project's entirety so it's easier to consume by the public, over my messy trial and error models in .ipynb

## Challenges and solutions

### Adjusting weight final model
I applied the same final model from waist, to weight. It didn't match the data as well as it did for waist, despite how similar I thought they would be. So these was some needed slight adjustements in hyperparameters.

### Model overgeneralizing prediction due to lifestyle
The overgeneralization was a common issue, and I decided to investigate it further by visualizing model prediction on the training data. The resulting graph shows very accurate predictions, until it overgeneralizes near the end portion of the prediction on training data.
![5b41f2f6-dde7-430e-9dbe-18957a36d45c](https://github.com/user-attachments/assets/f6857de9-8df4-4eba-aee9-5415ec5665e5)

I looked at the range of prediction where overgeneralizing starts to begin, in addition to x features of weight and caloric consumption
![0bf457fc-7674-4168-9edd-9354ca70c7f5](https://github.com/user-attachments/assets/c84cdda0-2107-4965-9f0c-498c4360979e)

We can see how the increase of caloric consumption and weight, is a big enough effect on the model's prediction. Reasoning behind this was, before then, I had a very sedentary lifestyle. The weight and calorie increase was when I changed into an active lifestyle. What likely happened was the model was learning on the patterns of my sedentary lifestyle, but was suddenly thrown at with active lifestyle data. It didn't have enough examples to figure out the patterns, and so it probably was safer to overgeneralize predictions for it.

Based on that, I tried training the model by only training on the active lifestyle portion of data. It's clear that training on the active lifestyle, helps model prediction on active lifestyle portion of data. But it still struggled with that last end of the data. My next thought is it could be a change in diet, where instead of bulking, I started to do some cutting during my active lifestyle. Other hypothesis is iit might not have been trained long enough to recognize all the patterns.
![5c231d43-bbf3-43b1-bba1-658e6b8619d7](https://github.com/user-attachments/assets/10d00200-5bba-4eea-887d-475ba8934ea4)

### Compounded prediction overestimates weight/waist
The difference values of weight and waist was actually a pretty useless factor to judge a model's accuracy. I figured adding all the prediction one by one from the first value of weight/waist measurment, and see what is the prediction's results, basing on measurment results.

Example of how the compounding loop would work: 
```
index 1 = start_value + weight_difference[1]
index 2 = index 1 + weight_difference[2]
index 3 = index 2 + weight_difference[3] and so on...
```

This is our biggest offender, the weight full data model, overpredicting the most. It also visually shows us, how the model overpredict likely due to the sedentary lifestyle in the training data. The model likely used the patterns in sedentary lifestyle, to make a prediction on the active lifestyle.
![f75fd59a-3ca9-4e29-9d42-fa21986fd9a3](https://github.com/user-attachments/assets/aea40ba5-09df-447d-9a3e-6f16f3970148)

### Wrong compounding prediction function!
But then also comes another issue with compounding prediction. This model simply recieves the true weight and waist value, everytime it makes its prediction. Realistically this is not feasible, as a compounding prediction problem, wouldn't have the future values of true weight and waist, after it's done a prediction. Think about it as a weather forecast, where you predict weather based on the prediction before since you don't have future data of said weather.

Another issue is that the difference prediction is not able to self correct itself, if it's previous prediction was too high or too low, since it's relying on true data, and following whatever seems logical/predictable for that value.

So in came a new function, where predictions made, will go back into the original X test data, allowing the model to make prediction from their previous prediction.

```python
def simulation_loop(model_weight, model_waist, norm_X, id_waist, id_weight, X_scaler, 
scaler_y_weight, scaler_y_waist, start_weight, start_waist, steps_ahead):
    """
    Forecasting weight and waist values through compounding predictions
    
    Parameters:
    model_weight, model_waist: trained models
    norm_X: standardized test set
    id_waist, id_weight: indexed location of where waist and weight is located in X_test dataset
    X_scaler, scaler_y_weight, scaler_y_waist: scalers for each respective datasets, to either fit or inverse transform when needed
    start_weight, start_waist: starting value of forecast
    steps_ahead: number of steps to predict/forecast
    """
    current_weight = float(start_weight) # make it float, as to not get rid of decimal predictions
    current_waist = float(start_waist)
    predictions = [] # keep new list for all predictions

    # unstandardize x_test
    X_test = X_scaler.inverse_transform(norm_X)    

    for i in range(steps_ahead):
        # prepare current index of x_test in loop to a new variable
        x = X_test[i].copy()

        # replace real measurements with predictions
        x[id_weight] = current_weight
        x[id_waist] = current_waist

        # reattach the current index row to the test dataset
        X_test[i] = x

        # scale test for model training
        norm_X = X_scaler.transform(X_test)

        # predict weight and waist
        weight_pred_scaled = model_weight.predict(norm_X, verbose=0)
        waist_pred_scaled = model_waist.predict(norm_X, verbose=0)

        # inverse scaling of y predictions (including the *100 done on data)
        weight_pred = scaler_y_weight.inverse_transform(weight_pred_scaled)[i][0]/100
        waist_pred = scaler_y_waist.inverse_transform(waist_pred_scaled)[i][0]/100

        # update values for next loop
        current_weight += weight_pred
        current_waist += waist_pred

        predictions.append((current_weight, current_waist))

    return predictions
```

### Using Quartile Loss to not over estimate values
I realized I had a bad understanding what quartile loss does, until this one graph told me everything I needed:
![prediction](https://github.com/user-attachments/assets/e5a5f04b-aacb-4944-a032-f8cef1ed986b)
Higher the quartile loss value, the higher it predicts based on range of data, and vice versa for lower. This gave me the knowledge of wehich hyperparamter to change to match test data better

Quartile Loss: 0.65
![1f7934aa-1977-42a3-82b4-917a0d0dc92f](https://github.com/user-attachments/assets/43ec2d03-d529-4d9d-8192-0ff1530b700c)
Quartile Loss: 0.7
![234259cf-4ea7-44bb-9861-31243b3aba4c](https://github.com/user-attachments/assets/bf9d084b-81ea-4c69-8baf-fc2142c9c6f7)

Thus finally ending up with my last prediction results of test for this project.

## Learnings and Insights

* **Lifestyle plays a big part in prediction:** Through the presentation of graphs, it's clear that physical activity/lifestyle has such a drastic impact on weight and waist measurment. This makes sense, as phsycial activity also plays a big part in burning calories. Given it's an input that was not indicated in the training data, the model did not expect this change in calorie, yet stability/tapering in weight and waist gain.
* **What is Quartile Loss?:** ChatGPT states that higher quartile loss values, result in more riskier prediction, which was what I want, due to overgeneralization. But what higher quartile loss really does, is predict the upper bound of the range of data given. Learning this, allowed me to adjust the hyperparameter so the compounding forecast values would match well with the truth value.
* **Visualize more than y data:** Sometimes, visualizing y data by itself may not always be conclusive of model accuracy. It's always good to use other ways to extrapolate the data if something isn't providing clear clues towards the model's performance.
* **Functions are time savers:** There were so many repeat of models and codes where I simply exchange between variable names. It would've been a big time saver if I tried to utilize functions more often in the workbook, in addition to less headache from constantly switching out variable names.

## Next steps

Move onto the next project or workbook
