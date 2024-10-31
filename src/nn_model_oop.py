
import math

def sigmoid(x):
    return 1/(1+math.exp(-x))

def softmax(x):
    exp_vec = list(map(lambda x: math.exp(x),x))
    sum_exp = sum(exp_vec)
    y = [v/sum_exp for v in exp_vec]
    return y

 
class simpleNN:
    # Initialize model parameters
    def __init__(self):
        self.w_1 = [[1., 1., 1.], [-1., -1., -1.]] 
        self.b_1 = [0,0,0]
        self.w_2 = [[1., 1.], [-1., -1.], [-1., -1.]]
        self.b_2 = [0,0]
    
    def forward(self,x):
        self.z = []
        self.h = []
        # First layer computer linear output z and sigmoid activation h
        for j in range(3):
           # Layer:1 linear output input (x) * weights vector (w)
           z_t=sum(self.w_1[i][j]*x[i] for i in range(2))+self.b_1[j]
           h_t = sigmoid(z_t)
           self.z.append(z_t)
           self.h.append(h_t)
        print(f"First linear output:{self.z}")
        print(f"Activation first layer:{self.h}")
        # Hidden layer output
        self.o = []
        for k in range(2):
            o_t = sum(self.w_2[j][k]*self.h[j] for j in range(3))+ self.b_2[k]
            self.o.append(o_t)
        print(f"Hidden layer linear output:{self.o}")
        # Compute softmax
        self.y = softmax(self.o)
        print(f"Softmax output:{self.y}")
        return self.y

        # Compute loss
    def calculate_loss(self,target_class):
        return -math.log(self.y[target_class])
    
    def backpropagation(self, x, target_class):
        # Step 1: Calculate dL/dy
        dl_dy = [0, 0]
        for i in range(2):
            if i == target_class:
                dl_dy[i] = -1 / self.y[i]
            else:
                dl_dy[i] = 0

        # Step 2: Calculate dy/do
        dy_do = [[0, 0], [0, 0]]
        for i in range(2):
            for j in range(2):
                if i == j:
                    dy_do[i][j] = self.y[i] * (1 - self.y[i])
                else:
                    dy_do[i][j] = -self.y[i] * self.y[j]

        # Step 3: Calculate dL/do
        dl_do = [0, 0]
        for k in range(2):
            for i in range(2):
                dl_do[k] += dl_dy[i] * dy_do[i][k]

        # Step 4: Calculate dL/dh (backprop from output to hidden layer)
        dl_dh = [0, 0, 0]
        for j in range(3):
            for k in range(2):
                dl_dh[j] += dl_do[k] * self.w_2[j][k]

        # Step 5: Calculate dL/dw2 (weights between hidden and output layer)
        dL_dw2 = [[0, 0], [0, 0], [0, 0]]
        for j in range(3):
            for k in range(2):
                dL_dw2[j][k] = dl_do[k] * self.h[j]

        # Step 5b: Calculate dL/db2 (biases for the output layer)
        dL_db2 = [dl_do[i] for i in range(2)]

        # Step 6: Calculate dL/dz (applying sigmoid derivative)
        dl_dz = [0, 0, 0]
        for j in range(3):
            dl_dz[j] = dl_dh[j] * self.h[j] * (1 - self.h[j])

        # Step 7: Calculate dL/dw1 (weights between input and hidden layer)
        dL_dw1 = [[0, 0, 0], [0, 0, 0]]
        for i in range(2):
            for j in range(3):
                dL_dw1[i][j] = dl_dz[j] * x[i]

        # Step 7b: Calculate dL/db1 (biases for the hidden layer)
        dL_db1 = [dl_dz[j] for j in range(3)]

        # Step 8: Calculate dL/dx (gradient with respect to inputs)
        dl_dx = [0, 0]
        for i in range(2):
            for j in range(3):
                dl_dx[i] += dl_dz[j] * self.w_1[i][j]

        # Return all gradients
        return dL_dw1, dL_dw2


if __name__ == "__main__":
    NN_model = simpleNN()
    x= [1,-1]
    y = NN_model.forward(x)
    loss = NN_model.calculate_loss(0)
    print(f"Current loss:{loss}")
    dw_1,dw_2 = NN_model.backpropagation(x,0)
    print(dw_1)
    print(dw_2)







        

        
