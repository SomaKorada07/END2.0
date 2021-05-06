## BACKGROUND AND VERY BASICS

## What is a neural network neuron?

Neural network neuron is like a storage. It is connected with input and output weights.  The computation of the product of weight and activation function happens outside the neuron. Activation functions could be tanh, sigmoid, ReLU, etc.



## What is the use of the learning rate?

Learning rate is used to make the model learn correctly. The amounts that the weights are updated during training is referred to as the step size or the “*learning rate*.” It is a hyperparameter in the training process. If a too large learning rate value is chosen, it could result in drastic updates to the weights resulting in divergent behaviors and on the other hand if a too small learning rate value is chosen, it could result in very small updates to weights resulting in getting stuck in a local minima.



## How are weights initialized?

Weights are randomly initialized from a Gaussian distribution. Another method is glorot (or Xavier) initializer which samples from a distribution but truncates the values based on the kernel complexity. In glorot initialization, each weight is initialized with a small Gaussian value with mean = 0.0 and variance based on the fan-in and fan-out of the weight; fan-in is number of input nodes and fan_out is number of output/hidden nodes.



## What is "loss" in a neural network?

Loss in a neural network is the difference between the true value and the model predicted value. It is calculated by different loss functions like Mean Squared Error (MSE), Binary Crossentropy, Categorical Crossentropy.



## What is the "chain rule" in gradient flow?

A gradient measures how much the output of a function changes if you change the inputs a little bit. The chain rule is used for calculating the derivative of composite functions. The chain rule can also be expressed in Leibniz's notation as follows:

If a variable **z** depends on the variable **y**, which itself depends on the variable **x**, so that *y* and *z* are dependent variables, then *z*, via the intermediate variable of *y*, depends on *x* as well. This is called the chain rule and is mathematically written as,

![img](https://miro.medium.com/max/181/1*7PhAl-QqLAc5vIOjEAQ-BA.png)

