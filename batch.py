import numpy as np

def forward(self, x, target_class):
    #  (hidden layer)
    self.z1 = np.dot(x, self.w_1) + self.b_1  # Where x is  (batch_size, input_size) *(input_size, hidden_size)
    self.h1 = sigmoid(self.z1)  # h1
    
    #  (output layer)
    self.z2 = np.dot(self.h1, self.v_1) + self.b_2  # h1 dimension  (batch_size, hidden_size) * (hidden_size, output size)
    self.y_pred = softmax(self.z2)  # y_pred is (batch_size, output_size)
    
    # Get the error for all samples
    self.loss = np.array([calculate_loss(self.y_pred[i], target_class[i]) for i in range(x.shape[0])])
    
    # Return y_pred and the mean of the loss for the batch
    return self.y_pred, np.mean(self.loss)

def backpropagation(self, x, target_class):
    m = x.shape[0]  # Number of samples in the batch
    
    # Step 1: Calculate dL/dy 
    y_true = np.zeros_like(self.y_pred)
    y_true[np.arange(m), target_class] = 1  # One-hot encoding, using np.arange let us one hot each row of the batch meaning that if 
    # if we have say batch equal 16 then we have a matrix of dimension (16,10), we will loop through each row and once in the row we set the column value to the 
    #position corresponding to the target class.
    dL_dy = self.y_pred - y_true  # Shape: (batch_size, output_size)
    
    # Step 2: Calculate dL/dv1. Let us make sure of dimensions
    dL_dv1 = np.dot(self.h1.T, dL_dy) / m  # (hidden_size, output_size)
    
    # Step 3: Calculate dL/dc1 
    dL_dc1 = np.sum(dL_dy, axis=0) / m  # (output_size,)
    
    # Step 4: Calculate dL/dh1 
    dL_dh1 = np.dot(dL_dy, self.v_1.T)  #(batch_size, hidden_size)
    
    # Step 5: Calculate dL/dz1 
    dL_dz1 = dL_dh1 * self.h1 * (1 - self.h1)  #  (batch_size, hidden_size)
    
    # Step 6: Calculate dL/dw1 
    dL_dw1 = np.dot(x.T, dL_dz1) / m  # (input_size, hidden_size)
    
    # Step 7: Calculate dL/db1 
    dL_db1 = np.sum(dL_dz1, axis=0) / m  #  (hidden_size,)
    
    # Update weights and biases
    learning_rate = 0.01
    self.w_1 -= learning_rate * dL_dw1
    self.b_1 -= learning_rate * dL_db1
    self.v_1 -= learning_rate * dL_dv1
    self.b_2 -= learning_rate * dL_dc1
