import numpy as np
#https://www.askpython.com/python/examples/principal-component-analysis

def PCA(X):
    # mean Centering the data  
    X_meaned = X - np.mean(X , axis = 0)
    # calculating the covariance matrix of the mean-centered data.
    cov_mat = np.cov(X_meaned , rowvar = False)
    #Calculating Eigenvalues and Eigenvectors of the covariance matrix
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
    #sort the eigenvalues in descending order
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    #similarly sort the eigenvectors 
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    #Transform the data 
    X_reduced = np.dot(sorted_eigenvectors.transpose(),X_meaned.transpose()).transpose()

    return (X_reduced,sorted_eigenvectors)

#Generate a dummy dataset.
np.random.seed(123)
X1 = np.random.randint(10,50,100).reshape(20,5) 
dat, v = PCA(X1)
