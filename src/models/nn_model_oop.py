
import math

def sigmoid(x) -> float:
    return 1/(1+math.exp(-x))

def softmax(x) ->list:
    exp_vec = list(map(lambda x: math.exp(x),x))
    sum_exp = sum(exp_vec)
    y = [v/sum_exp for v in exp_vec]
    return y

def calculate_loss(y_pred, target_class):
    return -math.log(y_pred[target_class])


# This function can be used to get the gradient with the correct decimal values quite unnecessary :)
def get_sig_weights(w_grad):
    w_sig = [[round(x,4) for x in sublist] for sublist in w_grad]
    return w_sig
def get_sig_bias(bias_grad):
    bias_sig = [round(x,4) for x in bias_grad]
    return bias_sig



# Default weights

#w_1 = [[1., 1., 1.], [-1., -1., -1.]] 
#v_1 = [[1., 1.], [-1., -1.], [-1., -1.]]
 
class simpleNN:
    # Initialize model parameters
    def __init__(self, w_1 = None,v_1 = None):
        self.w_1 = w_1 
        self.b_1 = [0,0,0]
        self.v_1 = v_1
        self.c_1 = [0,0]
    
    def forward(self,x,target_class):
        self.z = []
        self.h = []
        # First layer compute linear output z and sigmoid activation h
        for j in range(3):
           # index j loops over the three hidden layer units (3)
           # Layer:1 linear output W*x + b
           # Expected dimension (3,2)*(2,1) => (3,1)
           # We perform the operation for each unit and then append it to z
           # i iterates over the two input units
           z_t=sum(self.w_1[i][j]*x[i] for i in range(2))+self.b_1[j]
           h_t = sigmoid(z_t)
           self.z.append(z_t)
           self.h.append(h_t)
        #print(f"First linear output:{self.z}")
        #print(f"Activation first layer:{self.h}")
        # Hidden layer output
        self.o = []
        # Loop over the two k output units
        for k in range(2):
            # Use j to iterate over the three hidden layer units
            o_t = sum(self.v_1[j][k]*self.h[j] for j in range(3))+ self.c_1[k]
            self.o.append(o_t)
        #print(f"Hidden layer linear output:{self.o}")
        # Compute softmax
        self.y = softmax(self.o)
        #print(f"Softmax output:{self.y}")

        # Compute loss
        self.loss = calculate_loss(self.y,target_class)
        return self.y, self.loss


        # Compute loss
        #def calculate_loss(self,target_class):
        #return -math.log(self.y[target_class])
    
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
                dl_dh[j] += dl_do[k] * self.v_1[j][k]

        # Step 5: Calculate dL/dv2 (weights between hidden and output layer)
        dL_dv2 = [[0, 0], [0, 0], [0, 0]]
        for j in range(3):
            for k in range(2):
                dL_dv2[j][k] = dl_do[k] * self.h[j]

        # Step 5b: Calculate dL/dc2 (biases for the output layer)
        dL_dc2 = [dl_do[i] for i in range(2)]

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
        return (dL_dw1, dL_db1),(dL_dv2,dL_dc2)
    

# Run this function to run a whole forward and backpropagation step of the NN
def run_pass(w_1,v_1,x = [0,0], target_class = int) -> tuple:
    
    NN_model = simpleNN(w_1,v_1)
    #Predict
    y,loss = NN_model.forward(x,target_class)
    print(f"Predicted class:{y}")
    print(f"Current loss:{loss}")
    # Compute loss
    #loss = NN_model.calculate_loss(target_class)
    #print(f"Current loss:{loss}")
    dw_1,dv_1 = NN_model.backpropagation(x,target_class)
    # This prints are only to show in command line gradients with four significant decimal values, quite unnecessary :)
    print(f"W derivatives = {get_sig_weights(dw_1[0])}")
    print(f"b derivative:{get_sig_bias(dw_1[1])}")
    print(f"V derivatives: {get_sig_weights(dv_1[0])}")
    print(f"c derivative: {get_sig_bias(dv_1[1])} ")
    return dw_1,dv_1

if __name__ == "__main__":
    w_1 = [[1., 1., 1.], [-1., -1., -1.]]
    v_1 = [[1., 1.], [-1., -1.], [-1., -1.]]
    run_pass(w_1=w_1,v_1=v_1,x = [1,-1], target_class= 0)







        

        
