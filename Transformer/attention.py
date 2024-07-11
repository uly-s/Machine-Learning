from numpy import dot, array, zeros, matmul, random, exp, sqrt, max
import numpy 

# functions

def softmax(x):
    e_x = exp(x - max(x))
    return e_x / e_x.sum(axis=0)




# single attention head as a class
class Head:
    def __init__(self, dim_model, dim_k, dim_v):
        self.dim_model = dim_model
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.WQ = random.randn(dim_model, dim_k)
        self.WK = random.randn(dim_model, dim_k)
        self.WV = random.randn(dim_model, dim_v)
        self.WO = random.randn(dim_v, dim_model)
        self.softmax = Softmax()

    def forward(self, Q, K, V):
        Q = matmul(Q, self.WQ)
        K = matmul(K, self.WK)
        V = matmul(V, self.WV)
        scores = dot(Q, K.T) / sqrt(self.d_k)
        scores = softmax(scores)
        return matmul(scores, V)

    def backward(self, dY):
        dV = matmul(dY, self.WO.T)
        dScores = dot(dV, self.WV.T)
        dQ = matmul(dScores, self.WQ.T)
        dK = matmul(dScores, self.WK.T)
        return dQ, dK, dV

def compute_loss(outputs, targets):
    return numpy.mean((outputs - targets) ** 2)

def compute_loss_gradient(outputs, targets):
    return 2 * (outputs - targets) / outputs.size

def gradient_descent(model, inputs, targets, learning_rate, num_iterations):
    for i in range(num_iterations):
        # Compute the model's output
        outputs = model.forward(inputs)

        # Compute the loss
        loss = compute_loss(outputs, targets)

        # Compute the gradient of the loss with respect to the model's output
        dY = compute_loss_gradient(outputs, targets)

        # Compute the gradients of the model parameters
        dQ, dK, dV = model.backward(dY)

        # Update the model parameters according to their gradients
        model.WQ -= learning_rate * dQ
        model.WK -= learning_rate * dK
        model.WV -= learning_rate * dV

        # Print the loss every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")


# hypothetical test
# # Define the dimensions
# dim_model = 512
# dim_k = 64
# dim_v = 64

# # Create an instance of the Head class
# head = Head(dim_model, dim_k, dim_v)

# # Create some random input data
# Q = np.random.randn(dim_model, dim_k)
# K = np.random.randn(dim_model, dim_k)
# V = np.random.randn(dim_model, dim_v)

# # Pass the input data through the forward method
# output = head.forward(Q, K, V)

# # Print the output
# print("Output:", output)

# # Create some random gradient data
# dY = np.random.randn(dim_v, dim_model)

# # Pass the gradient data through the backward method
# dQ, dK, dV = head.backward(dY)

# # Print the gradients
# print("dQ:", dQ)
# print("dK:", dK)
# print("dV:", dV)

# # Compute the loss
# targets = np.random.randn(*output.shape)
# loss = compute_loss(output, targets)

# # Print the loss
# print("Loss:", loss)

# # Compute the gradient of the loss
# dY = compute_loss_gradient(output, targets)

# # Print the gradient of the loss
# print("dY:", dY)


