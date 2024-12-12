import numpy as np

# Part 3. Question 6 Batch version
# Define activation functions and loss calculation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # stabilize with max subtraction
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def calculate_loss(y_pred, target_class):
    return -np.log(y_pred[np.arange(y_pred.shape[0]), target_class])

class SimpleNN:
    def __init__(self, input_size=784, hidden_size=300, output_size=10):
        # Initialize weights and biases
        # Initially we multiply by 0.01 this to avoid large activations, smaller gradients
        self.w_1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b_1 = np.zeros(hidden_size)
        self.v_1 = np.random.randn(hidden_size, output_size) * 0.01
        self.b_2 = np.zeros(output_size)
        
    def forward(self, x, target_class):
        #  (hidden layer)
        self.z1 = np.dot(x, self.w_1) + self.b_1  # Where x is  (batch_size, input_size) *(input_size, hidden_size)
        self.h1 = sigmoid(self.z1)  # h1
    
        #  (output layer)
        self.z2 = np.dot(self.h1, self.v_1) + self.b_2  # h1 dimension  (batch_size, hidden_size) * (hidden_size, output size)
        self.y_pred = softmax(self.z2)  # y_pred is (batch_size, output_size)
    
        # Get the error for all samples
        self.loss = calculate_loss(self.y_pred, target_class) 
    
        # Return y_pred and the mean of the loss for the batch
        return self.y_pred, np.mean(self.loss)

    def backpropagation(self, x,target_class, l_rate = 0.01):
        # Number of samples
        m = x.shape[0]
        
        # Step 1: dL/dy
        y_true = np.zeros_like(self.y_pred)
        y_true[np.arange(m), target_class] = 1
        dL_dy = self.y_pred - y_true
        
        # Step 2: dL/dv1, dL/dc1
        dL_dv1 = self.h1.T.dot(dL_dy) / m
        dL_dc1 = np.sum(dL_dy, axis=0) / m
        
        # Step 3: dL/dh1
        dL_dh1 = dL_dy.dot(self.v_1.T)
        
        # Step 4: dL/dz1 (sigmoid derivative)
        dL_dz1 = dL_dh1 * self.h1 * (1 - self.h1)
        
        # Step 5: dL/dw1, dL/db1
        dL_dw1 = x.T.dot(dL_dz1) / m
        dL_db1 = np.sum(dL_dz1, axis=0) / m
        
        # Update weights and biases
        self.w_1 -= l_rate * dL_dw1
        self.b_1 -= l_rate * dL_db1
        self.v_1 -= l_rate * dL_dv1
        self.b_2 -= l_rate * dL_dc1




