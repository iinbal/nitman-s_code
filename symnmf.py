import numpy as np
import pandas as pd
import sys
import symnmf_c as symnmf

np.random.seed(1234)

def init_h(n, k, m):
    """
    Initialize the H matrix.

    Args:
        n (int): Number of data points (rows).
        k (int): Number of clusters (columns).
        m (float): Mean value of the norm matrix.

    Returns:
        numpy.ndarray: Randomly initialized H matrix of shape (n, k).
    """
    return np.random.uniform(
        low=0.0,
        high=2 * np.sqrt(m/k),
        size=(n, k)
    )

def m_val(norm_matrix):
    """
    Returns the mean value of the norm matrix.

    Args: 
        norm_matrix (numpy.ndarray): Norm matrix. 

    Returns: 
        float: Mean of norm_matrix.
    """        
    m = np.mean(norm_matrix)
    return m
 
def ex_funcs(goal_str, p_matrix, n, k):
    """
    Returns a matrix based on requested operation.

    Args: 
        goal_str (str): Type of matrix to compute based on user's input.
        p_matrix (list[list[float]]): Input points array.
        n (int): Number of data points. 
        k (int): Number of clusters.

    Returns: 
        list[list[float]]: requested matrix.
    """        
    match goal_str:
        case "sym":
            a_mat = symnmf.sym(p_matrix)
            return a_mat
        case "ddg":
            d_mat = symnmf.ddg(p_matrix)
            return d_mat
        case "norm":
            norm_mat = symnmf.norm(p_matrix)
            return norm_mat
        case _:
            norm = symnmf.norm(p_matrix)
            norm_mat = np.array(norm)       
            m =  m_val(norm_mat)
            h_in = init_h(n, k, m)
            h_mat = symnmf.symnmf(h_in.tolist(), norm, n, k) 
            return h_mat 

def print_d(mat, n):
    """
    Print the Diagonal Degree matrix given its diagonal values.

    Args: 
        mat (np.ndarray): 1D array of Diagonal values.
        n (int): Number of data points.

    Returns: 
        None
    """        
    for i in range(n):
        for j in range(n):
            if i == j:
                print(f"{mat[i]:.4f}", end="")
            else:
                print("0.0000", end="")
            if j < n - 1:
                print(",", end="")
        print()  

def main():
    try:
        args = sys.argv
        k = int(args[1])
        goal = args[2]
        filename = args[3]
        file = pd.read_csv(filename, header=None)
        points_arr = file.to_numpy()
        n = points_arr.shape[0]
        points = points_arr.tolist()
        res = ex_funcs(goal, points, n, k)
        res_mat = np.array(res)
        if(goal == "ddg"):
            print_d(res_mat, n)
        else:
            for row in res_mat:
                print(','.join(f"{val:.4f}" for val in row))
    except Exception as e:
        print("An Error Has Occurred")

if __name__ == "__main__":
    main()