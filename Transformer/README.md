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

------ 

Update these notes for that full OpenAI chat, then start on backprop.

After updating your model with the full encoder / decoder architecture.

------

## Encoder / Decoder
The transformer architecture is split into encoder / decoder modules or layers - I like modules because each one is composed of layers composed of more layers.

The standard architecture is such

### Encoder
An encoder consists of N (typically N=6) identical layers each consisting of a multi-headed attention sub-layer and a feed-forward sub layer. The multi headed attention layer is as described above and in ```attention.py```. The ff layer is basically just a vanilla feed foward 2 layer deep neural net with  ReLu activation function. 

The multi-head attention layer computes the output matrix as described elsewhere and the ff layer takes each vector of the inputted matrix as input (position wise input) to compute the input for the next layer or decoder.

When asked about the feed-forward nets and why they're necessary, chatgpt said "The FFN introduces non-linear transformations, enabling the model to capture complex patterns." I'm not sure exactly what that means but should later.

Additionally layer normalization and residual connections are applied at each sub layer- layer normalization is quite straight forward and more or less just 'squishing' signal / input such that the signal is more 'leveled out'. 

Residual connections are somewhat counter-intuitively named because they are described as 'skipping layers' making you think of a neural-architecture, when in actually the math more closely resembles 'signal accumulation' rather than residual connections.

Both techniques help to alleviate the vanishing gradients problem of neural networks.

### Decoder
The decoder is almost identical to the encoder - made up of N (typically N = 6) identical layers each composed of 2 sub layers - except instead of being the regular multi-headed attention layer of the encoder it uses masked multi headed attention.

Masked multi headed attention is basically just taking an upper triangle segment of the input matrix - masking some of the input such that the decoder can't look into the future and take in input tokens it shouldn't know about. Sans masking its just regular multi headed attention.

Like the encoder for each attention sub layer there is a corresponding ff sub layer applying ReLu in between its two sub-sub-sub layers. Providing the same services and functionality as in the encoder. Followed by normalization and residuals after each sub layer just as in the encoder.

## Output
After the decoder layers the signal undergoes a few more transformations.

"The final steps involve converting the decoder's output into a probability distribution over the target vocabulary, from which the final tokens are generated." 

First the final output of the decoder layers is passed into a linear (fully connected) layer that 'projects the decoders output to the size of the target vocabulary.' 

The output from which is passed through a softmax function to convert it into a probability distribution over the target vocabulary for each token in the sequence. This probability distribution indicates the likelihood of each token in the vocabulary being the next token in the output sequence.

I'm not sure of the exact nature of this last part - except that it feels like the output sequence is "filtered through" the proceeding probability distribution where each token is a like a ball in a pachinko machine.

Finally Argmax (inferencing) - the token with the highest probability is selected as the next token.




