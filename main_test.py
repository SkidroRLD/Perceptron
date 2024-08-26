import data_gen
import perceptron
import torch
import torch.optim as optim
import numpy
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
torch.cuda.set_device(0)

def criterion(y_pred, y_true):
    # Binary Cross Entropy Loss
    # y_pred: predicted probabilities, y_true: true labels (0 or 1)
    
    # Compute the negative log likelihood loss using binary cross-entropy formula
    # (y * log(y_pred) + (1 - y) * log(1 - y_pred))
    loss = -1 * (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    
    # Calculate the mean loss over the batch
    mean_loss = torch.mean(loss)
    
    return mean_loss

if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

def main():
    x, y = data_gen.gen_data()
    perceptronObj = perceptron.model(1, 2, 1)
    # print(perceptronObj(x))
    optimizer = optim.SGD(perceptronObj.parameters(), lr=0.01) #optimiser
    losses = []
    for epoch in tqdm(range(0,5000)):
        total_loss = 0
        for x1, y1 in zip(x, y):
            optimizer.zero_grad()

            yhat = perceptronObj(x1)
            # Calculate the loss between the predicted output (yhat) and the actual target (y)
            loss = criterion(yhat, y1)

            # Backpropagation: Compute gradients of the model parameters with respect to the loss
            loss.backward()

            # Update the model parameters using the computed gradients
            optimizer.step()

            # Accumulate the loss for this batch of data
            total_loss += loss.item()

        # Append the total loss for this epoch to the cost list
            losses.append(total_loss)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch} done!")  # Print status after every 1000 epochs

        # Plot the result of the function approximator
            predicted_values = perceptronObj(x).cpu().detach().numpy()
            print(predicted_values)
            plt.plot(x.cpu().numpy(), predicted_values)  # Plot the predicted values
            plt.plot(x.cpu().numpy(), y.cpu().numpy(), 'm')  # Plot the ground truth data (Y)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend(['Predicted', 'Ground Truth'])
            plt.title(f'Epoch {epoch} - Function Approximation')
            plt.savefig('plot ' + str(epoch) + '.jpg')
    
    # plt.plot(cost, marker='o', linestyle='-', color='b', label='Training Loss')

    # # Set labels and title
    # plt.xlabel('Epochs')
    # plt.ylabel('Cross Entropy Loss')
    # plt.title('Training Progress - Cross Entropy Loss')

# Add grid for better readability
# plt.grid(True)

# # Show legend
# plt.legend()

# # Display the plot
# plt.show()

if __name__ == '__main__':
    main()