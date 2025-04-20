---
layout: post
tags: Weight/Waist_prediction_project
todolist: Title, Overview, Projects and Tasks, Challenges and Solutions, Learnings and Insights, Next Steps, Reflections
---

# Week 6 & 7

## Relevant Work
[Weight and waist workbook](https://tenatic-x.github.io/_projects/weight%20and%20waist%20workbook.html)
[Weight and waist pdf report](https://tenatic-x.github.io/_projects/workbook%20report.pdf)

## Overview/Tasks

Continuing the weight and waist project, now working on TensorFlow models, and experimenting with hyperparameters, layers, activations etc.

## Challenges and solutions

This was another struggle, trying to perfect or at least find suitable hyperparameters, functions, and layers for the model to get an accurate prediction of my weight and waist changing overtime, day by day. Again will summarize the 50+ different variations of the models I've tested.
* Changed up delta value in huber (aka the weight of how much to use mae or mse as loss)
* My new training data, has a time sequence, keeping a window size of days in the training set data (aka previous days information). Tried the model on the original data without this time sequence, did not predict well. That extra time sequence on the training data really does make a difference!
* Played with the number of layers and its neurons from time to time
* Used different learning rate function changers, e.g. `cyclic_lr`, `weighted_log_cosh` to `quantile_loss`
* Standardized my training and test data
* Way too many small changes that made little to no improvement.

So far here's the resulting graph output for the model, once it's trained on waist and weight data:
![502215ca-052a-4c8a-94e0-77cee10fa201](https://github.com/user-attachments/assets/a087d0c7-2df4-46a9-afdc-32a0e11ab105)

![0abde423-643e-4819-a7bb-b1ff9df84573](https://github.com/user-attachments/assets/4bcf68be-6dd2-4de2-895a-57a6b2a81e5d)

### Data leakage!
As I continued training the model, something starts to eerily take shape in the graph of prediction vs truth labels... The prediction looks super similar to the true values, only that it has a slight delay on it's prediction.
![db120335-a349-4919-be80-5743f2f2d1bc](https://github.com/user-attachments/assets/a1642a26-665e-47a7-9aea-faf78b35f3a1)
ChatGPT created code for me to check the lag:
```python
def cross_correlation(true_data, predicted_data):
    correlation = np.correlate(true_data, predicted_data, mode='full')
    lag = np.argmax(correlation) - len(true_data) + 1
    return lag

Optimal lag is: -2
```

This wasn't seen from the original training data, so it must've been from the time sequence data we've formulated. I decided to check the code with chatgpt, and it found the issue! It was when I was appending the y data.

```python
# OLD code
def create_sequences(X, y, window_size):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i+window_size])
        ys.append(y[i+window_size])
    return np.array(Xs), np.array(ys)

# NEW code
def create_sequences(X, y, window_size):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i+window_size])
        ys.append(y[i]) # This part
    return np.array(Xs), np.array(ys)
```

The differece at the y append between `i` and `i+window_size`, is the former only returns the current index of value y. The latter, returns the index value, that has the `window_size` added upon it (e.g. i=2, window_size=7, then y=2+7). What it does, it has the model predicting future values, rather than current values. This makes it easy for the model to now predict, as the values are leaked in `X data`, due to `y data` being in the future, aka accessing data that it's not supposed to have!

With this small change, the model now predicts properly.

### Organization between weight, waist, X, y, train, test, and time sequence data

Now I'm reaching the home stretch of this project, there are a lot of variables I have to name for each one I'll be using. And it's been really annoying to keep track of the names in what order, where they're supposed to be, and whether I've put them in the wrong order or duplicated them, then forgot to change their name. Since working on the waist data solely, I didn't really name my model to `waist` accordingly, but more akin to `model` or `X_data`, in order to save time and got comfy with just training on waist. This did create a bit of a headache, now doing the weight training, and sometimes accidentally overwriting the weight data with the waist, which lead to retraining.

### Oversized and lengthy workbook

I realized how long this workbook has gotten, especially with the number of model trainings, accompanied by loss graphs over epoch, and the prediction against truth value graphs. It's been frustrating to scroll around for the right sequence of code, or information I was looking for.

### Model still has trouble fitting onto test data

Despite running 99 variations of the model, I've continued to struggle to get really strong results in my predictions. It's hard to get my model's to take risks, and I know, that if the model just predicted 0 all the way, it would've done better in terms of accuracy than if they risked, and tried to predict weight value. So I just pushed up the risk to where I felt was okay, and gauge by looking at graph output.

## Learnings and Insights

* **Organization of variable names no matter what:** It was a bit of an awakening at how important naming variables really are, and it must be of priority no matter where you are at your project. I was less careful when working into the waist data, that I negated that I'll be doing the weight data, in the name of efficiency and also my lack of short term memory. If I have trouble recalling and stating the variables to my code, then the readers of this code will have a complete anneurism reading my shit!
* **Splitting your workbook:** It might be a good idea, to instead split my workbook into steps of the data science process. Aka workbook on cleaning data and feature engineering, and another workbook trying out Tensorflow models on the data.
* **Loss value in training, don't always translate into testing:** One thing I noted during training, is how easily some of the models got in reaching a loss of 0, way before 100 epochs. I thought that any epochs beyond that will cause overfitting on the test data. Once I tested on the smaller epochs, the model just heavily underfitted on the test. My hypothesis is the model may learn especially quickly with the training data, but needed time to actually learn the nuances of the data, so it could generalize better for test data
* **Model trouble with fitting to test:** Another thing, is the model, despite a lot of tests, don't fit properly, especially with most of the outlier values. Pondering about why this is happening, and how it's not reaching the same conclusions I've found on practice readily prepeared data, there are multiple reasons:
* * There's more things that influence your weight and waist fluctuations day by day. Things like water weight, sodium - as that holds in water inside your body, the timing of my meals, the timing when I weighed myself, etc.
  * Too many outliers/missing data. There are days where I forget to record my weight, and just set it to 0 for convenience. This also leads to massive outliers, due to a period of time when data was not recorded, but also because of other things that may influence my weight outside of calories and macronutrients.
  * Maybe I just don't have enough data. The prediction of weight and waist may be too complex, and a few years of data point might not be enough to fully capture patterns for the model.


## Next steps

Work on the last steps of graph creations and visuals of the final models, but also continued fixing up whatever needs to be done
