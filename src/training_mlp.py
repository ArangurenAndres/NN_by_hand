from utils.process_data import load_synth, norm_data, load_config, plot_loss
from models.nn_model_oop import simpleNN
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_model(nn_model, epochs=1, lr=0.01, batch_size=1, visualize  = True, save_result = False, filename = None):
    #1. Process data
    train_set,val_set,_ = load_synth(num_train=60_000, num_val=10_000, seed=0)
    x_train,y_train = train_set
    x_val,y_val = val_set
    #3.After splitting the data apply data normalization to a range [0,1]
    # We do this step after data splitting to avoid test_data leakage
    x_train = norm_data(x_train)
    x_val = norm_data(x_val)
    ## Set network weights
    # This list will store the losses through the defined epochs
    run_loss = []
    for i in tqdm(range(epochs)):
        epoch_loss = []
        # Since we are iterationg one by one data point this is SGD if we want minibatch we would have to get the loss for the batch
        # and update the weights respectively
        for x_,y_ in zip(x_train,y_train):
            # 4. Get prediction and loss
            _,loss = nn_model.forward(x_,y_)
            # We append the loss for the data point to the iteration loss
            epoch_loss.append(round(float(loss),4))
            # 6. Compute gradient to update weights and biases (W,b and V,c)
            dw_1,dv_1 = nn_model.backpropagation(x_,y_)
            dw, db = dw_1
            dv, dc = dv_1
            #print(f"Gradient w: {dw} | Gradient b:{db}")
            #print(f"Gradient w: {dv} | Gradient b:{dc}")
            #7. Update weights for each observation SGD w <- w - lr*dl/dw
            # Update w,b
            #print(f"Weights W before: {nn_model.w_1}")
            nn_model.w_1 -= lr*np.array(dw)
            nn_model.b_1 -= lr*np.array(db)
            #print(f"Weights W after update: {nn_model.w_1}")

            # Update v,c
            #print(f"Weights V before: {nn_model.v_1}")
            nn_model.v_1 -= lr*np.array(dv)
            nn_model.c_1 -= lr*np.array(dc)
            #print(f"Weights V after update: {nn_model.v_1}")
        loss_mean = round(np.mean(epoch_loss),4)
        run_loss.append(loss_mean)
        #print(run_loss)
    if visualize:
        plot_loss(run_loss,save = save_result,filename=filename)
    else:
        pass


    return run_loss




if __name__ =="__main__":
    #  1. Define hyper parameters
    config = load_config()

    epochs = config["epochs"]
    lr = config["lr"]
    batch_size = config["batch_size"]
    print(f"Running training with the follwoing parameters:")
    print(f"Epochs: {epochs}")
    print(f"Learning_rate: {lr}")  
    print(f"Batch size: {batch_size}")
    # 2. Set the model

    #For the model set the desired weights for now we will use random distributed values
    w_1 = np.random.randn(2,3)
    v_1 = np.random.randn(3,2)
    nn_model = simpleNN(w_1=w_1,v_1=v_1)

    # 3. Train model
    run_loss = train_model(nn_model, epochs=epochs, lr=lr, batch_size=batch_size, save_result = True, filename = "prueba.png")
    loss_correct = [float(x) for x in run_loss]
    print(f"Running loss:{loss_correct}")










        









