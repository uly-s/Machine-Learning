Hold on, this is important. Please help me understand better "The basic idea is to adjust each weight in the opposite direction of its gradient, which decreases the output error. " walk me through the logic step by step starting with what a gradient is in this context.

Sure, let's break it down:

What is a gradient? In the context of neural networks, a gradient is a vector that contains the partial derivatives of a function with respect to its inputs. In simpler terms, each element in the gradient vector indicates how much the output of the function would change if the corresponding input was increased by a tiny amount. In the case of a neural network, the function is the network's loss function, and the inputs are the weights and biases of the network.

Why do we want to compute the gradient? The goal of training a neural network is to find the weights and biases that minimize the loss function. The gradient gives us a direction in the high-dimensional space of weights and biases in which the loss function increases the fastest. Therefore, if we want to decrease the loss function as quickly as possible, we should move in the opposite direction of the gradient. This is the idea behind gradient descent.

How do we compute the gradient? The process of computing the gradient of the loss function with respect to the weights and biases of a neural network is called backpropagation. Starting from the output layer, we use the chain rule of calculus to compute the gradients layer by layer, moving backwards through the network.

How do we use the gradient to update the weights and biases? Once we have computed the gradient, we can use it to update the weights and biases. The simplest way to do this is to subtract a small fraction of the gradient from the weights and biases. This is called a gradient descent step. The size of the step is controlled by a parameter called the learning rate. If the learning rate is too large, the updates may overshoot the minimum of the loss function; if it's too small, the training process may be very slow.

Here's a simple pseudocode example of a gradient descent step:

This process is repeated many times, using the training data to compute the loss function and its gradient. Over time, this should cause the weights and biases to converge to values that minimize the loss function, and thus make the network's predictions as accurate as possible.

------

I can clearly enough picture this in my head, but I can't clearly visualize from input -> feedfoward all the way to this step and back. But I don't feel far off. 

Looking at it again the math is... really not that crazy and worth sitting down to learn itself. I could use a calm refresher. 

If I can do 'deeper' and understand intuitively how the derivatives work in the minute case and 'wide' in understanding (and picturing) the data / numeric flow from start to finish, I will feel perfectly employable as a machine learning engineer.

(Also finally learning pytorch would help... ya know a lot.)

Also update this repo from master -> main