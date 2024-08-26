import torch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
torch.cuda.set_device(0)

def gen_data():
    X = torch.arange(-30,30,1).view(-1,1).type(torch.FloatTensor).to(device)

    Y = torch.zeros(X.shape[0]).to(device)

    # Assign label 1.0 to elements in Y where the corresponding X value is less than or equal to -10
    Y[X[:, 0] <= -10] = 1.0

    # Assign label 0.5 to elements in Y where the corresponding X value falls between -10 and 10 (exclusive)
    Y[(X[:, 0] > -10) & (X[:, 0] < 10)] = 2

    # Assign label 0 to elements in Y where the corresponding X value is greater than 10
    Y[X[:, 0] >= 10] = 3
    print(X,Y)
    return X, Y.view(-1,1)

if __name__ == '__main__':
    gen_data()