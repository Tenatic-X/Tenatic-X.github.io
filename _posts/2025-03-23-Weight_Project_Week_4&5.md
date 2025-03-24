---
layout: post
tags: Weight/Waist_prediction_project
todolist: Title, Overview, Projects and Tasks, Challenges and Solutions, Learnings and Insights, Next Steps, Reflections
---

## Overview/Tasks

Continuing the weight and waist project, now working on TensorFlow models, and experimenting with hyperparameters, layers, activations etc.

## Challenges and solutions

This has been a very challenging 2 weeks, with so many trials and errors, but way too many dead ends. To summarize it all, I will bullet point the steps of me trying to find the most ideal model:
* Using `ReLU` layers, and changing neuron density on different layers.
* I then spent time training on `Waist` over `Weight`, as trying to make 2 predictions on a model thats trained on 2 separate times, seem to make it predict the same regardless. We'll come back to the `Weight` data, after we find ideal `Waist` prediction due to how similar the data is.
* Played with `Adam` learning rate, and changing different optimizer, learning rate doesn't seem to make much effect, and `Adam` was the best optimizer overall.
* I started using `Recurrent Neural Networks (RNNs)`, should've stopped using it after seeing constant predictions of 0, but somehow got way too invested in this specific activation. So used `LSTM` and `GRU`.
* Increased Epoch during training, as the model needed more time to train to see patterns and gain valuable insight on its predictions/loss.
* Changing loss variable from `MSE` to `Huber`, hoping the model takes more risk as it has been too conservative with its predictions, and displaying 0.
* Train up to 1,000 Epochs, the model seems to start flatten out and stop improving at random points during training. When checking the graph, improvements can be seen, but there are just areas where it refuses to make predictions, and stay at 0.

This is an example of the Loss, getting stuck at some point in the training:

  ![1ef41c4c-1f73-4c0d-9fbb-8e7b4b28ad94](https://github.com/user-attachments/assets/c00cee56-ba61-48e1-bd4b-3a2fe5cd0724)

And this is an example of my model, where it just predicts sometimes, then it doesn't in certain areas:

![8c1d849b-8147-44ec-978a-d19238500c7f](https://github.com/user-attachments/assets/e2b2b3a2-deae-4f04-8ccf-90afdf444841)

* Many failed attempts later, I asked ChatGPT for a model example, and heavily improved the model's training. I had problem with the model's loss flattening out randomly, and this model from ChatGPT doesnt flatten!

This is it's Loss overtime:

![2170f64d-3af4-413b-a928-f9797b921d5a](https://github.com/user-attachments/assets/e683c241-1e38-4790-8534-8de92bdd7ab1)

And it's prediction:

![1d79a184-e01f-4c81-811d-5c7fdcd4b9d3](https://github.com/user-attachments/assets/f7358b22-39ad-4ff0-80a7-920b9b04fcb3)


This was my best performing code:
```python
# set seed
tf.random.set_seed(35)

expandX_train = np.expand_dims(Xwaist_sequence_train, axis=-1)  # Expanding last axis for feature dimension
expandX_test = np.expand_dims(Xwaist_sequence_test, axis=-1)


# create model including relu and linear
modelwaist_34 = keras.Sequential([
    layers.GRU(64, activation='tanh', return_sequences=True, input_shape=(expandX_train.shape[1], expandX_train.shape[2])),
    layers.GRU(128, activation='tanh', return_sequences=True),
    layers.GRU(64, activation='tanh', return_sequences=False),  # Last LSTM layer should have return_sequences=False
    layers.BatchNormalization(),
    layers.Dropout(0.2),


    layers.Dense(128),
    layers.LeakyReLU(alpha=0.1), 
    layers.Dense(64),
    layers.LeakyReLU(alpha=0.1),
    layers.Dense(32),
    layers.LeakyReLU(alpha=0.1),
    layers.Dense(16),
    layers.LeakyReLU(alpha=0.1),
    layers.Dense(8),
    layers.LeakyReLU(alpha=0.1),
    
    layers.Dense(1, activation='linear')
])

reduce_lr = ReduceLROnPlateau(monitor='loss',  # Track loss (can also use 'val_loss')
                              factor=0.8,      # Reduce LR by half when triggered
                              patience=10,     # Wait 10 epochs before reducing
                              min_lr=1e-10)     # Do not reduce below this

modelwaist_34.compile(optimizer=Adam(learning_rate=0.001),
                loss=Huber(delta=93.2),
                metrics=['mae'])

hist_34 = modelwaist_34.fit(expandX_train, ywaist_sequence_train, epochs=1000, verbose=0, callbacks=[reduce_lr])
```

And this was ChatGPT's model:
```python
expandX_train = expandX_train.reshape(expandX_train.shape[0], -1)


# Define the model
gptmodel = keras.Sequential([
    layers.Dense(128, activation=None, input_shape=(expandX_train.shape[1],)),  
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.2),

    layers.Dense(64, activation=None),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.2),

    layers.Dense(32, activation=None),
    layers.BatchNormalization(),
    layers.ReLU(),

    layers.Dense(1)  # Output layer for regression
])

# Define optimizer with a fixed learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.0005)

# Learning rate scheduler
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=1e-6)

# Compile the model
gptmodel.compile(optimizer=optimizer, loss=Huber(delta=50), metrics=['mae'])

# Train the model
gpthistory = gptmodel.fit(expandX_train, ywaist_sequence_train, 
                    epochs=1000, verbose=1, callbacks=[reduce_lr])
```
Look how simple the design of the model is! Especially the layers! No need for all the `LSTM` or `GRU` bullshit. Just `ReLU` then normalizing and doing dopouts, and it does it's job a million times better.


## Learnings and Insights

* **The sunk cost fallacy:** The more I worked with `RNNs`, the more I was invested into this activation, and wanting to make it work. I was blind sighted that this activation was going to make my model prediction even better, but was secretely sabotaging me the whole time. The more I worked on `RNN` layers, the more I wanted it to work for me. It took a second opinion from ChatGPT, and a step back by looking at my older `TensorFlow` model trainings to realize that I've hitting a dead end this whole time. Simple `ReLU` layers were already making predictions on every point they got, and never suffered the `0` prediction areas of the `RNN` graphs.
* **Complex doesn't mean better:** Despite `RNNs` being something that looks at a passing number of data points, e.g. the previous 7 days of my weight/waist data to make its calculation, this can actually be a source of difficulty/complexity to the model predicting it. I'm still not sure why it likes to flatten out training at random given points, could be because there's not enough data, too many 0s, or values that vary too much. Regardless, I should've tried exploring the simpler options, then slowly move to more complex options.
* **Once in a while, try high Epochs on model:** You never really know the extent of a model's performance, unless you give it a hail mary once in a while. It's only when I decided to train on `1000 epochs` on the model, do I really see the limits of `RNNs` on my model, and how effective `ReLU` layers can be, with some normalization and dropout layers.

## Next steps

Further tweak my model to align with what ChatGPT has shat out, further understand why does `RNN` not work on my data, then continue to improve upon my model to match or out perform ChatGPT's model.
