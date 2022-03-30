import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import copy


NUM_INPUT = 784
NUM_OUTPUT = 10

# Function to Split the data
def splitData(data, factor=0.8):
    return np.split(data,[int(factor * len(data))])

# Function to perform ReLU
def ReLu(z):
    return np.maximum(0,z)

# Function to perform the differentiation of Relu
def dReLu(z):
    z[z<=0] = 0
    z[z>0] = 1
    return z

# Unpack a list of weights and biases into their individual np.arrays.
def unpack (weightsAndBiases, NUM_HIDDEN=10, NUM_HIDDEN_LAYERS=3):
    # Unpack arguments
    Ws = []

    # Weight matrices
    start = 0
    end = NUM_INPUT*NUM_HIDDEN
    W = weightsAndBiases[start:end]
    Ws.append(W)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN*NUM_HIDDEN
        W = weightsAndBiases[start:end]
        Ws.append(W)

    start = end
    end = end + NUM_HIDDEN*NUM_OUTPUT
    W = weightsAndBiases[start:end]
    Ws.append(W)

    Ws[0] = Ws[0].reshape(NUM_HIDDEN, NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN, NUM_HIDDEN)
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN)

    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN
    b = weightsAndBiases[start:end]
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN
        b = weightsAndBiases[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weightsAndBiases[start:end]
    bs.append(b)

    return Ws, bs

def forward_prop (x, y, weightsAndBiases, alpha=0, NUM_HIDDEN=10, NUM_HIDDEN_LAYERS=3, loss_type='regularized'):
    alpha=0.0
    zs=[]
    hs=[]
    hs.append(x) #Inserting the input layer into h
    Ws, bs = unpack(weightsAndBiases, NUM_HIDDEN, NUM_HIDDEN_LAYERS)

    z=np.dot(Ws[0],x)+np.atleast_2d(bs[0]).T
    zs.append(z)
    h=ReLu(z)
    hs.append(h)
    
    for i in range(1,NUM_HIDDEN_LAYERS):
        z=np.dot(Ws[i],hs[i])+np.atleast_2d(bs[i]).T
        zs.append(z)
        h=ReLu(z)
        hs.append(h)
    
    z=np.dot(Ws[-1],hs[-1])+np.atleast_2d(bs[-1]).T
    zs.append(z)

    yhat=np.exp(zs[-1])/np.sum(np.exp(zs[-1]),axis=0)[None,:]
    yhat_log = np.log(yhat, out=np.zeros_like(yhat), where=(yhat!=0))
    
    # Calculating Cross Entropy Loss on Validation
    start=0
    end=len(weightsAndBiases)-((NUM_HIDDEN_LAYERS*NUM_HIDDEN)+NUM_OUTPUT)
    if(loss_type!='unregularised'):
        loss= -(np.sum(y*yhat_log))/(np.shape(y)[1]) + alpha*np.dot(weightsAndBiases[start:end].T,weightsAndBiases[start:end])/(2*np.shape(y)[1])
    else:
        loss= -(np.sum(y*yhat_log))/(np.shape(y)[1])

    return loss, zs, hs, yhat
   
def back_prop (x, y, weightsAndBiases, alpha=0, NUM_HIDDEN=10, NUM_HIDDEN_LAYERS=3):
    alpha=0.0
    loss, zs, hs, yhat = forward_prop(x, y, weightsAndBiases, alpha, NUM_HIDDEN, NUM_HIDDEN_LAYERS)
    Ws, bs = unpack(weightsAndBiases, NUM_HIDDEN, NUM_HIDDEN_LAYERS)
    
    dJdWs = []  # Gradients w.r.t. weights
    dJdbs = []  # Gradients w.r.t. biases

    g= (yhat-y)/np.shape(y)[1]
    # TODO
    for i in range(NUM_HIDDEN_LAYERS, -1, -1):
        dW=np.dot(g,hs[i].T) + alpha*Ws[i]/np.shape(y)[1]
        dJdWs.insert(0,dW)
        db=np.sum(g,axis=1)
        dJdbs.insert(0,db)
        if(i!=0):
            g= np.dot(Ws[i].T,g)*dReLu(zs[i-1])

    # Concatenate gradients
    return np.hstack([ dJdW.flatten() for dJdW in dJdWs ] + [ dJdb.flatten() for dJdb in dJdbs ]) 

# Performs a standard form of random initialization of weights and biases
def initWeightsAndBiases (NUM_HIDDEN=10, NUM_HIDDEN_LAYERS=3):
    Ws = []
    bs = []

    np.random.seed(0)
    W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN)
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN)
        bs.append(b)

    W = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    
    return np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])

def SGD(X, Y, weightsAndBiases, epoch, alpha, eps, mb, num_hidden, num_hidden_layers, loss_type='regularized'):
    N=np.shape(X)[1]
    trajectory=[]
    # Stochastic Gradient Descent
    for i in range(epoch):
        for j in range(int(N/mb)):
            X_mb=X[:,j*mb:(j+1)*mb]
            Y_mb=Y[:,j*mb:(j+1)*mb]
            dweightsAndBiases= back_prop(X_mb, Y_mb, weightsAndBiases, alpha, num_hidden, num_hidden_layers)
            weightsAndBiases-= eps*dweightsAndBiases
        # Performing deepcopy to avoid the changing of trajectory list when updating the weights and biases
        temp=copy.deepcopy(weightsAndBiases)
        trajectory.append(temp)            

        # Performing forward propogation to determine training loss
        tr_loss, tr_zs, tr_hs, tr_yhat= forward_prop (X, Y, weightsAndBiases, alpha, num_hidden,num_hidden_layers, loss_type)

        # Performing prediction on training set
        trY_pred=np.argmax(tr_yhat,axis=0)
        trY_orig=np.argmax(Y,axis=0)
        accuracy = np.sum(trY_pred==trY_orig)/np.shape(Y)[1]
        print("Training loss and Accuracy at Epoch ",i+1,": %.2f" %tr_loss,", %.2f" %(accuracy*100), "%")
                            
    return weightsAndBiases, trajectory

def findBestHyperparameters(trainX,trainY):
    X_tr, X_va= splitData(trainX.T)
    X_tr= X_tr.T
    X_va= X_va.T
    Y_tr, Y_va= splitData(trainY.T)
    Y_tr= Y_tr.T
    Y_va= Y_va.T
    
    # Best Hyperparameters
    # NUM_HIDDEN=[50]
    # NUM_HIDDEN_LAYERS=[3]
    # Epoch=[35]
    # Alpha=[1]
    # Eps=[0.01]
    # Mb=[16]

    # For Grid Search of Hyperparameters
    NUM_HIDDEN=[30,40,50]
    NUM_HIDDEN_LAYERS=[3,4,5]
    Epoch=[20,35,75]
    Alpha=[1,0.1]
    Eps=[0.001, 0.005, 0.01]
    Mb=[16,64,128]
    
    best_HP=np.array(6)
    min_loss=np.Inf
    
    print("----------------------------------------------------------------------")
    print("Performing Grid Search for Combinations of Hyperparameters:")
    print("----------------------------------------------------------------------")

    for num_hidden in NUM_HIDDEN:
        for num_hidden_layers in NUM_HIDDEN_LAYERS:
            for epoch in Epoch:
                for alpha in Alpha:
                    for eps in Eps:
                        for mb in Mb:
                            #Initializing weights and biases
                            weightsAndBiases=initWeightsAndBiases(num_hidden,num_hidden_layers)
                            
                            #Defining the hyperparameter
                            HP= [num_hidden, num_hidden_layers, epoch, alpha, eps, mb]
                            print("\nTraining on Hyperparameters:",HP)
                            
                            #Performing SGD and extracting the weight and biases
                            updated_weightsAndBiases = SGD(X_tr, Y_tr, weightsAndBiases, epoch, alpha, eps, mb, num_hidden, num_hidden_layers)[0]
                            
                            #Calculating the Validation Loss and Accuracy
                            va_loss, va_zs, va_hs, va_yhat= forward_prop (X_va, Y_va, updated_weightsAndBiases, alpha, num_hidden,num_hidden_layers)
                            vaY_pred=np.argmax(va_yhat,axis=0)
                            vaY_orig=np.argmax(Y_va,axis=0)
                            accuracy = np.sum(vaY_pred==vaY_orig)/np.shape(Y_va)[1]
                            print("Validation loss and Accuracy: %.2f" %va_loss,", %.2f" %(accuracy*100), "%")

                            #Checking if this hyperparameter set is better than the best uptill now
                            if(va_loss<min_loss):
                                min_loss=va_loss
                                best_HP= HP
                                print("Hyperparameters Updated")
                                np.save('wb_train.npy', updated_weightsAndBiases)
                            else:
                                print("Hyperparameters Not Updated")
                            print("Best Hyperparameter uptill now:",best_HP)
    return best_HP

def train (trainX, trainY, testX, testY):
    #Find the best hyperparameter by performing SGD on each
    best_HP=findBestHyperparameters(trainX,trainY)

    num_hidden, num_hidden_layers, epoch, alpha, eps, mb = best_HP
    print("\n----------------------------------------------------------------------")
    print(" Grid Search Completed")
    print("----------------------------------------------------------------------")
    print(" Results after performing Grid Search:")
    print(" Best Hyperparameters:")
    print("   Hidden Inputs in Each Layer= ",num_hidden)
    print("   Hidden Layers= ",num_hidden_layers)
    print("   Epochs= ",epoch)
    print("   Alpha= ",alpha)
    print("   Learning Rate= ",eps)
    print("   Mini Batch Size= ", mb)

    print("\n----------------------------------------------------------------------")
    print("Training on Training + Validation Dataset:")
    print("----------------------------------------------------------------------")
    weightsAndBiases=initWeightsAndBiases(num_hidden,num_hidden_layers)

    #Performing training on validation and test data set
    weightsAndBiases, trajectory = SGD(trainX, trainY, weightsAndBiases, epoch, alpha, eps, mb, num_hidden, num_hidden_layers,'unregularized')

    # Calculating Loss and Accuracy on Test Dataset
    te_loss, te_zs, te_hs, te_yhat = forward_prop (testX, testY, weightsAndBiases, alpha, num_hidden,num_hidden_layers)
    testY_pred=np.argmax(te_yhat,axis=0)
    testY_orig=np.argmax(testY,axis=0)
    accuracy = np.sum(testY_pred==testY_orig)/np.shape(testY)[1]

    print("\n----------------------------------------------------------------------")
    print("Performance Evaluation")
    print("----------------------------------------------------------------------")
    print("Testing loss and Accuracy: %.2f" %te_loss,", %.2f" %(accuracy*100), "%\n")

    return weightsAndBiases, trajectory

def plotSGDPath (trainX, trainY, ws):
    # TODO: change this toy plot to show a 2-d projection of the weight space
    # along with the associated loss (cross-entropy), plus a superimposed 
    # trajectory across the landscape that was traversed using SGD. Use
    # sklearn.decomposition.PCA's fit_transform and inverse_transform methods.

    def toyFunction (x1, x2):
        xy=[x1,x2]
        xy_inverse=pca.inverse_transform(xy)
        loss= forward_prop (trainX, trainY, xy_inverse, 1, 50, 3, loss_type='unregularized')[0]
        return loss

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Standardizing the Data
    scaler= StandardScaler()
    scaler.fit(ws)
    ws=scaler.transform(ws)

    # Performing PCA
    pca=PCA(n_components=2)
    xy=pca.fit_transform(ws)

    #Creating the mesh grid for surface plot
    x=xy[:,0]
    y=xy[:,1]
    axis1 = np.linspace(np.min(x),np.max(x), 50)
    axis2 = np.linspace(np.min(y),np.max(y), 50)
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Zaxis = np.zeros((len(axis1), len(axis2)))

    for i in range(len(axis1)):
        for j in range(len(axis2)):
            #Calcualting the Loss Value
            Zaxis[i,j] = toyFunction(Xaxis[i,j], Yaxis[i,j])

    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)

    # Now superimpose a scatter plot showing the weights during SGD.
    X=[]
    Y=[]
    Z=[]
    for i in range(0,len(x)-1):
        #Creating 10 points between each epoch
        tempx=np.linspace(x[i],x[i+1],10)
        tempy=np.linspace(y[i],y[i+1],10)
        for j in range(10):
            X.append(tempx[j])
            Y.append(tempy[j])
            #Calculating the Loss value
            Z.append(toyFunction(tempx[j], tempy[j])) 
    
    ax.scatter(X,Y,Z, color='r')
    plt.show()

if __name__ == "__main__":
    # TODO: Load data and split into train, validation, test set
    X_data = np.reshape(np.load("fashion_mnist_train_images.npy"), (-1, 28*28))/255
    X_tr, X_va= splitData(X_data)
    trainX= X_data.T
    X_tr= X_tr.T
    X_va= X_va.T

    trainY = np.load("fashion_mnist_train_labels.npy")
    trainY=(np.eye(NUM_OUTPUT)[trainY]).T
    Y_tr, Y_va= splitData(trainY.T)
    Y_tr= Y_tr.T
    Y_va= Y_va.T
    
    # Loading Test Dataset
    X_te = np.reshape(np.load("fashion_mnist_test_images.npy"), (-1, 28*28))/255
    testX= X_te.T
    testY = np.load("fashion_mnist_test_labels.npy")
    testY = (np.eye(NUM_OUTPUT)[testY]).T

    # Initialize weights and biases randomly
    weightsAndBiases = initWeightsAndBiases()

    # Perform gradient check on 5 training examples
    print(" The Gradient check on 5 training examples gives the error as shown below:")
    print(scipy.optimize.check_grad(lambda wab: forward_prop(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), wab)[0], \
                                    lambda wab: back_prop(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), wab), \
                                    weightsAndBiases))

    # Training and Testing
    weightsAndBiases, trajectory = train(trainX, trainY, testX, testY)

    np.save('trajectory.npy', trajectory)
    np.save('wb_test.npy', weightsAndBiases)
    
    trajectory=np.load("trajectory.npy")

    # Reducing data size for Plotting SGD trajectory 
    x=splitData(trainX.T, 0.9)[1]
    y=splitData(trainY.T,0.9)[1]
    
    # Plot the SGD trajectory
    plotSGDPath(x.T, y.T, trajectory)


    