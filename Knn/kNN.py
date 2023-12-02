import numpy as np

def kNN(k, X, labels, y):

    m = X.shape[0] # number of training examples
    n = X.shape[1] # number of attributes

    # distance from the point y to each element of the training set. 
    # Check that its size equals the number of training points
    distances = np.linalg.norm(X - y, axis=1)
    

    # Sort distances and re-arrange labels based on the distance of the instances
    idx = distances.argsort()
    labels = labels[idx]

    # retrieve the index of the k closest points :
    c = np.zeros(max(labels)+1)
	
    # Compute the class labels of the k nearest neigbors
    for i in range(k):
        c[labels[i]] += 1

    # Return the label with the largest number of appearances
    # rk, if you chose a too small K, it can happen that there is a tie between some classes. 
    # In this case np will select the first label of the tie
    label = np.argmax(c)
        
    return label
