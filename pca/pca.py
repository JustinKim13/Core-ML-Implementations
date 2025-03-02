from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # load dataset per instructions
    x = np.load(filename)
    
    # center the dataset
    mean = np.mean(x, axis=0)
    
    # subtract our mean from every value in x
    x_cent = x - mean    
    return x_cent

def get_covariance(dataset):
    # dataset -> centered dataset from load_and_center_dataset()
    
    n = dataset.shape[0] # number of rows 
    return (1 / (n - 1)) * np.dot(np.transpose(dataset), dataset) # plug into equation

def get_eig(S, k):
    # S -> covariance matrix
    # k -> # of largest eigenvalues and corresponding eigenvectors to return
    
    n = S.shape[0] # num of rows 
    eigenvalues, eigenvectors = eigh(S, subset_by_index=[n - k, n - 1]) # gets us the largest k eigenvalues
    
    # now lets reverse the order of eigenvalues and eigenvectors
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1] # only reverse the columns, keep all rows 
    
    # return largest m eigenvalues of S in descending order as a m x m diagonal matrix, followed by the eigenvectors
    
    # lets first create our eigenvalues matrix, Lambda
    Lambda = np.diag(eigenvalues)
    
    # now we can return the full thing
    return Lambda, eigenvectors
    
def get_eig_prop(S, prop):
    # S -> covariance matrix
    # prop -> proportion of variance in the dataset explained by the ith eigenvector
    
    n = S.shape[0] # num of rows
    
    # trace -> sum of its eigenvalues
    # to get trace, we're going to need to get initial eigenvalues first
    eigenvalues, eigenvectors = eigh(S)
    
    # n_greater = num i with prop of variance greater than the provided prop
    n_greater = np.sum(eigenvalues / sum(eigenvalues) > prop)
    
    # use eigh() again, this time with subset_by_index with our new n_greater variable
    eigenvalues, eigenvectors = eigh(S, subset_by_index=[n - n_greater, n - 1])
    
    # reverse eigenvalues and eigenvectors 
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1] # only reverse the columns, keep all rows
    
    # convert eigenvalues to our Lambda matrix
    Lambda = np.diag(eigenvalues)
    
    return Lambda, eigenvectors

def project_and_reconstruct_image(image, U):
    # image -> d x 1 column vector
    # U -> eigenvectors from get_eig() or get_eig_prop()
    
    # essentially going to do U @ (transpose(U) @ image)
    
    # project the image onto the PCA subspace
    proj = np.transpose(U) @ image # perfrom matrix multiplication on transposed U and image
    
    # reconstruct image using projection and return it
    re_image = U @ proj # perfrom matrix multiplication to reconstruct
    return re_image

def display_image(im_orig_fullres, im_orig, im_reconstructed):
    # im_orig_fullres -> original high res image
    # im_orig -> original image
    # im_reconstructed -> reconstructed image
    
    # reshape images to be 60 x 50
    im_orig_reshaped =  im_orig.reshape(60, 50)
    im_reconstructed_reshaped =im_reconstructed.reshape(60, 50)
    
    # create a figure with one row of three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(9,3), ncols=3)
    
    # set titles
    ax1.set_title("Original High Res")
    ax2.set_title("Original")
    ax3.set_title("Reconstructed")
    
    # display images on correct axes
    ax1.imshow(im_orig_fullres.reshape(218, 178, 3), aspect='equal') # reshape before displaying
    im2 = ax2.imshow(im_orig_reshaped, cmap='gray', aspect='equal')
    im3 = ax3.imshow(im_reconstructed_reshaped, cmap='gray', aspect='equal')

    # create a colorbar for ax2 and ax3 on right
    fig.colorbar(im2, ax=ax2)
    fig.colorbar(im3, ax=ax3)

    # moved to end
    fig.tight_layout()

    return fig, ax1, ax2, ax3

def perturb_image(image, U, sigma):
    # image -> orignal image; to be flattened
    # U -> matrix of top m eigenvectors
    # sigma -> std deviation of Gaussian noise added
    
    # get our PCA projection of image
    alpha = np.transpose(U) @ image 
    alpha_shape = alpha.shape
    
    # perturbation should be from a Gaussian distribution with mean 0 and sd sigma
    perturbation = np.random.normal(loc=0, scale=sigma, size=alpha_shape) 
    
    # now we can just reconstruct our perturbed image
    x_perturbed = U @ (alpha + perturbation)
    
    return x_perturbed

