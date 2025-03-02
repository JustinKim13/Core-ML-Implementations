import csv
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def load_data(filepath : str) -> List[Dict[str, str]]:
    '''
    Takes in a string with a path to a CSV file and returns the data points as a
    list of dictionaries.
    '''
    country_list = [] # list to append our dicts to
    with open(filepath, newline='') as file: # open our file, formatting to remove new lines
        reader = csv.DictReader(file)
        for row in reader: # for each row (country), we'll append a dict of the row
            country_list.append(dict(row))
            
    return country_list 

def calc_features(row : Dict[str, str]) -> np.ndarray:
    '''
    Takes in one row dictionary from the data loaded from the previous function,
    calculates the corresponding feature vector for that country as specified, and returns it as a NumPy
    array of shape (9,). The dtype of this array should be float64.
    '''
    features = [ # create a list with the float vals of all our features from row item 
        float(row["child_mort"]),
        float(row["exports"]),
        float(row["health"]),
        float(row["imports"]),
        float(row["income"]),
        float(row["inflation"]),
        float(row["life_expec"]),
        float(row["total_fer"]),
        float(row["gdpp"])
    ]
    
    return np.array(features, dtype=np.float64) # make sure datatype is correct

def hac(features : np.ndarray) -> np.ndarray:
    '''
    Performs complete linkage hierarchical agglomerative clustering on the countries
    using the (x1, . . . , x9) feature representation and returns a NumPy array representing the clustering.
    '''
    n = len(features) # we're going to reuse this a lot
    
    # assign each data point a unique index from 0 to n-1
    clusters = {i : [i] for i in range(n)}
    
    # create an (n − 1) × 4 array or list 
    Z = np.zeros((n - 1, 4))
    
    # create dist matrix using hashmap
    dist_matrix = {}
    for r in range(n): # iterate through to set dist_matrix values
        for c in range(r + 1, n): # since a distance matrix is symmetric, we only need to take either the top or bottom triangle, and the diagonal is always 0
            dist_matrix[(r, c)] = np.linalg.norm(features[r] - features[c]) # distance function, put directly into matrix
    
    # perform n-1 merges
    for r in range(n - 1):
        # find 2 closest clusters
        dist = float("inf") # since we're taking the min, we'll start with this at infinity
        c1, c2 = None, None # closest cluster pair
        
        for key, val in dist_matrix.items():
            if val < dist: # if current dist is less than min dist, update dist and cluster papir
                dist = val
                c1, c2 = key # store the pair        
                
        # now that we found the min dist, we can merge the 2 clusters, placing them in their correct matrix spots according to insturctions
        Z[r, 0] = c1
        Z[r, 1] = c2
        Z[r, 2] = dist
        Z[r, 3] = len(clusters[c1]) + len(clusters[c2]) # add cluster sizes together to get size of our newly merged cluster
        
        # create new cluster
        new_cluster = n + r # n gets us to the end, then r represents the current merge we're on
        
        # merge clusters
        clusters[new_cluster] = clusters[c1] + clusters[c2] 
        del clusters[c1] # remove old clusters
        del clusters[c2]
        
        # find old distances from dist matrix to delete
        old_dist = []
        for key in dist_matrix:
            if c1 in key or c2 in key:
                old_dist.append(key)
                
        # remove keys to delete
        for key in old_dist:
            del dist_matrix[key]
            
        # find distance for new cluster using complete linkage
        for cluster in list(clusters.keys()):
            if cluster != new_cluster: # don't want to check distance with itself
                max_dist = float("-inf") # start w -inf since we're taking the max
                for i in clusters[new_cluster]: # check each point in new cluster
                    for j in clusters[cluster]: # check each point in existing cluster
                        cur_dist = np.linalg.norm(features[i] - features[j])
                        if cur_dist > max_dist:
                            max_dist = cur_dist
                 # make sure we're setting it as the smaller val fist, then larger
                dist_matrix[(min(new_cluster, cluster), max(new_cluster, cluster))] = max_dist
                    
    # convert z to np and return  
    return Z   
        
def fig_hac(Z: np.ndarray, names: List[str]) -> plt.Figure:
    '''
    Visualizes the hierarchical agglomerative clustering of the countries feature
    representation.
    '''
    fig = plt.figure() # as per instructions
    dendrogram(Z, labels=names, leaf_rotation=90) # rotate labels sideways so they all fit, like shown in the example
    plt.tight_layout() # helps with formatting
    plt.show() # show our plot, and return the fig
    
    return fig

def normalize_features(features: List[np.ndarray]) -> List[np.ndarray]:
    '''
    Takes a list of feature vectors and computes the normalized values.
    The output should be a list of normalized feature vectors in the same format as the input. 
    '''
    # create an array out of our features
    features_matrix = np.array(features) 
    
    # create arrays for our means and stds
    mean = np.mean(features_matrix, axis=0)
    std = np.std(features_matrix, axis=0)
    
    # create normalized matrix, applying correct z-score formula
    normalized_matrix = (features_matrix - mean) / std
    
    # convert back to same format as input
    normalized_list = []
    for i in range(normalized_matrix.shape[0]): # iterate over each row
        normalized_list.append(normalized_matrix[i, :]) # append each row to list
    
    # return in correct format
    return normalized_list

def main() -> None:
    # provided testing code: 
    data = load_data("Country-data.csv")
    features = [calc_features(row) for row in data]
    names = [row["country"] for row in data]
    features_normalized = normalize_features(features)
    np.savetxt("output.txt", features_normalized)
    n = 20
    Z = hac(features[:n])
    fig = fig_hac(Z, names[:n])
    plt.show()

if __name__ == "__main__":
    main()