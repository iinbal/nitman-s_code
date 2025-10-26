import numpy as np
import pandas as pd
import sys
import symnmf
from kmeanshw1 import final_clusters
from sklearn.metrics import silhouette_score

#gets file name, returns a tuple of (X, n) when X=2D numpy array of data points, 
# n = num of points
def get_points(f_name): 
    """
    Load data points from input file.

    Args:
        f_name (str): name of the file containing the data points.

    Returns:
            points (numpy.ndarray): 2D array of shape (n, d) with the read data points.
            num_p (int): Number of data points (rows).
    """
    dfp = pd.read_csv(f_name, header=None)
    points= dfp.to_numpy()
    num_p = points.shape[0]
    return points, num_p


def get_sym_culsters(goal, points_arr, n, k): 
    """
    Run SymNMF clustering and return cluster assignments vector.

    Args:
        goal (str): Clustering goal ("symnmf").
        points_arr (numpy.ndarray): 2D array of shape (n, d) with input data points.
        n (int): Number of data points.
        k (int): Number of clusters.

    Returns:
        numpy.ndarray: 1D array of length n where the i-th entry is the cluster assignment index of the i-th point.
    """
    points = points_arr.tolist()
    res = symnmf.ex_funcs(goal, points, n, k)
    res_arr = np.array(res, dtype=float)
    cluster_ind = np.argmax(res_arr, axis=1)
    return cluster_ind

def main():
    try:
        args = sys.argv
        goal = "symnmf"
        k = int(args[1])
        filename = args[2]
        X, n= get_points(filename)
        Y_sym = get_sym_culsters(goal, X, n, k)

        # Calculate silhouette scores for SymNMF clustering
        sym_s_score = silhouette_score(X, Y_sym)
        Y_kmeans = final_clusters(X, k)

        # Calculate silhouette scores for KMeans clustering
        kmeans_s_score = silhouette_score(X, Y_kmeans)
        print(f"nmf: {sym_s_score:.4f}")
        print(f"kmeans: {kmeans_s_score:.4f}")
    except Exception as e:
        print("An Error Has Occurred")

if __name__ == "__main__":
    main()    