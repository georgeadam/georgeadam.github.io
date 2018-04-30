
# Universal Approximation Theorem - Part 3

## Wrapping Up

You've made it to the final part of this blog series! It's been a fun journey seeing just how flexible neural networks are - so flexible that they have an unlimited ability to overfit. The first two posts in the series were a bit negative and lacked the optimism that is typically associated with any kind of ML. Luckily, today gives us the chance to think on the bright side. That is to say, we now get to do some experimenting and see how far that takes us in an attempt to fit difficult functions while being able to generalize.


```python
%matplotlib notebook

import numpy as np
import matplotlib.pyplot as plt

from utils.analysis import get_weights, predict_targets
from utils.data import create_data, cubic_function, sin_function, quartic_function

from models.custom import create_custom_model, create_periodic_model
from models.callbacks import CustomLRScheduler
from keras.optimizers import Adam


```


```python
fig = plt.figure(figsize=(8, 8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()

fig.show()
fig.canvas.draw()
```


```python
def train_inner(model, weight, x_train, y_train, x_test, y_test, callbacks):
    model.compile(optimizer=Adam(lr=1),
                  loss="mean_squared_error",
                  metrics=["mean_squared_error"])

    layer = model.get_layer("periodic")
    layer.set_trainable_weight(weight)

    model.fit(x_train, y_train, epochs=25, validation_data=[x_test, y_test], verbose=False, callbacks=callbacks)
    print(layer.get_weights())
    
    pred = model.predict(x_train)
    ax.clear()
    ax.plot(x_train.reshape(-1), y_train.reshape(-1), color="b")
    ax.plot(x_train.reshape(-1), pred.reshape(-1), color="r")

    fig.canvas.draw()
    
    return pred


def train_outer(model, epochs, x_train, y_train, x_test, y_test, callbacks):
    weights = ["periodic_amplitude", "periodic_shift", "periodic_periodicity"]
    preds = []

    for i in range(epochs):
        print("Epoch: {}".format(i))
        
        for weight in weights:
            print("Updating weight: {}".format(weight))
            pred = train_inner(model, weight, x_train, y_train, x_test, y_test, callbacks)
            preds.append(pred)

    return model, preds 


def train_basic(model, epochs, x_train, y_train, x_test, y_test):
    model.compile(optimizer=Adam(lr=1),
                 loss="mean_squared_error",
                 metrics=["mean_squared_error"])
    preds = []
    
    for i in range(epochs):
        model.fit(x_train, y_train, epochs=25, validation_data=[x_test, y_test], verbose=False)
        pred = model.predict(x_train)
        preds.append(pred)
    
    return model, preds
```


```python
x_train, y_train, x_test, y_test = create_data(-5, 5, -7, 7, 1000, sin_function, [4, 20, 0.5, 0], add_noise=False)

lr_scheduler = [CustomLRScheduler()]
model = create_periodic_model(0.5, 0.00005)
model_iterative, preds_iterative = train_outer(model, 10, x_train, y_train, x_test, y_test, lr_scheduler)

model = create_periodic_model(0.5, 0.00005)
model_all, preds_all = train_basic(model, 10, x_train, y_train, x_test, y_test)
```


```python
from utils.plot import animated_training_plot
animated_training_plot(x_train, y_train, preds_iterative, "training_clip_iterative.gif", fps=15)
animated_training_plot(x_train, y_train, preds_all, "training_clip_all.gif", fps=15)
```


```python

```
