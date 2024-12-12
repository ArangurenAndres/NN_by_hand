import numpy as np
from utils.data import load_mnist, load
import matplotlib.pyplot as plt
from models.nn_model_vector import SimpleNN, calculate_loss
from utils.process_data import plot_loss
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

def visualize_images(x):
    # Assuming x_train is already loaded and contains flattened 28x28 images
    # Reshape the flattened images into their original shape (28x28)
    x_train = x.reshape(-1, 28, 28)

    # Select 9 random indices from the training set
    random_indices = np.random.choice(x_train.shape[0], 9, replace=False)

    # Plot the 9 random images in a 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_train[random_indices[i]], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# Step 3: Plot the Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.title("Confusion Matrix", fontsize=16)
    plt.show()



def get_accuracy(model,x,y):
    correct = 0
    y_pred,_ = model.forward(x,target_class=None)
    y_pred_classes = []
    for y_,y_true in zip(y_pred,y):
        # Find the argmax meaning the index in y_pred that maximizes the probability of the class
        y_class = np.argmax(y_)
        y_pred_classes.append(y_class)
        # If the index that has the maximum probability is equal to the target class then will be classified as correct classification
        if y_class==y_true:
            correct+=1
            # divide the number of correct classified samples over the length of all samples
    accuracy = correct/len(y)
    print(f"Model accuracy: {accuracy}")
    return accuracy, y_pred_classes

# Step 3: Plot the Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.title("Confusion Matrix", fontsize=16)
    plt.show()
        
    #fin the argmax
    


def plot_iterations(x,it):
    loss = np.array(x)
    mean_loss = np.mean(loss,axis=0)
    std_loss = np.std(loss,axis=0)
    print(len(mean_loss))
    x = [i for i in range(len(loss[0]))]
    # Plot mean loss with standard deviation as shaded area
    plt.plot(x, mean_loss, label='Average Loss', color='blue')
    plt.fill_between(x, mean_loss - 3.0*std_loss, mean_loss + 3.0*std_loss, color='blue', alpha=0.2, label='Standard Deviation')
    # Labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Average Loss with st.deviation over 5 Epochs | iterations = {it}')
    plt.legend()
    plt.grid()
    plt.show()



# Training loop
def train_for_epochs(model, x, y, x_val,y_val, lr=0.01 ,batch_size=16,epochs=3):
    # Initialize the trianing loss as empty list, it will store the loss values for each epoch
    total_train_loss = []
    # Initialize the validaiton loss as empty list
    total_val_loss = []
    for epoch in tqdm(range(epochs)):
        print(f"Epoch {epoch + 1}/{epochs}...")
        num_batches = x.shape[0] // batch_size
        print(f"Number of batches/epoch: {num_batches}")
        batch_loss = []
        for i in tqdm(range(num_batches)):
            x_batch = x[i*batch_size:(i+1)*batch_size]
            y_batch = y[i*batch_size:(i+1)*batch_size]
            
            # Forward and backpropagation
            _, loss = model.forward(x_batch, y_batch)
            model.backpropagation(x_batch, y_batch,l_rate = lr)
            
            batch_loss.append(loss)  # Append the losses for each batch
            
        epoch_loss = np.mean(batch_loss)
        total_train_loss.append(epoch_loss)
        #Perform the forward pass and calculate loss for the validation data set
        _, val_loss = model.forward(x_val, y_val)
        # We append the mean loss 
        total_val_loss.append(np.mean(val_loss))

    #Correct the data type
    total_train_loss = [float(x) for x in total_train_loss]
    total_val_loss = [float(x) for x in total_val_loss]

        #print(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss}")
    return total_train_loss, total_val_loss

def run_iterations(x_train,y_train,x_val,y_val,it=2):
    train_loss_it = []
    val_loss_it = []
    i=0
    while i<=it:
        print(f"Iteration : {i+1}")
        # In each iteration the model weights and biases will be initialized with new random values
        model = SimpleNN()
        train_loss, val_loss = train_for_epochs(model, x_train, y_train, x_val,y_val, batch_size=32,epochs=5)
        train_loss_it.append(train_loss)
        val_loss_it.append(val_loss)
        i+=1
    return train_loss_it, val_loss_it





if __name__ == "__main__":
    #When set final to true we will run the training and canonical test dataset
    (x_train, y_train), (x_test, y_test), _ = load_mnist(final = True,flatten=True)
    visualize_images(x_train)
    # 1. Initialize the model
    # Part 3 Question 5
    #model = SimpleNN()
    #2. Train model
    #train_loss, val_loss = train_for_epochs(model, x_train, y_train, x_test,y_test,lr=0.01, batch_size=32,epochs=5)
    # plot_loss(train_loss,val_loss, visualize=True, save = True, filename = "final_test.png")
    #accuracy, y_pred_classes = get_accuracy(model,x_test,y_test)

    # Plot confusion matrix
    # Define the class names for MNIST (0-9)
    #class_names = [str(i) for i in range(10)]

    # Plot the confusion matrix
    #plot_confusion_matrix(y_test, y_pred_classes, class_names)

    
    







    # #2. Set desired list of l_rates
    # #The list of learning rates follows a logarithmic scale
    #lr_list = [0.0001, 0.0005, 0.001, 0.005]
    #lr_list = [0.01,0.001,0.05,0.0001]
    #train_losses= []
    # Loop through different learning rates and train the model
    #epochs =5
    #for lr_it in lr_list:
    #    train_loss, val_loss = train_for_epochs(model, x_train, y_train, x_val, y_val, lr=lr_it, batch_size=1, epochs=epochs)
    #    train_losses.append(train_loss)  # Store the train loss for this learning rate

    #Plot the train loss for each learning rate
    #plt.figure(figsize=(10, 6))

    # Ensure that train_losses[i] is a list of losses over epochs
    #for i, lr_it in enumerate(lr_list):
    #    sns.lineplot(x=range(0, epochs), y=train_losses[i], label=f'lr={lr_it:.0e}', lw=2)
    # Add axis labels and title
    #plt.xlabel('Epochs', fontsize=12)
    #plt.ylabel('Training Loss', fontsize=12)
    #plt.title('Training Loss vs Epochs for Different Learning Rates', fontsize=14)
    # Add a legend
    #plt.legend(title="Learning Rates", title_fontsize=12, fontsize=11)
    # Show grid and layout adjustments
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()
    

    # Plot the loss of each batch against the timestep
    #plot_loss(train_loss,val_loss,save=True,filename="loss_5_epochs_test.png")

    #Question 7.2
    #it = 10
    #train_loss,val_loss = run_iterations(x_train,y_train,x_val,y_val,it=it)
    #print(len(train_loss))
    #print(len(val_loss))
    #plot_iterations(train_loss,it)




