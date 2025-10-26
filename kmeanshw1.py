import math
import numpy as np

EPSILON = 1e-4

#All code from HW1 which was irrelevant was removed

#calculates the distance between two points
def getDistance(point1, point2):
    """
    Compute Euclidean distance between two points.

    Args:
        point1 (list[float]): First point.
        point2 (list[float]): Second point.

    Returns:
        float: Euclidean distance.
    """
    d = len(point1)
    dist = 0
    for i in range(d):
        diff = float(point1[i])-float(point2[i])
        dist += diff*diff
    return math.sqrt(dist)

#updates the centroids after we insert all points       
def update_center(clusters):
    """
    Compute new centroids after we insert all points.

    Args:
        clusters (list[list[list[float]]]): List of clusters, each has a list of points.

    Returns:
        list[list[float]]: Updated centroids.
    """
    d = len(clusters[0][0])
    centers = []
    for cluster in clusters:
        up_center= []
        for i in range(d): 
            s = 0
            for point in cluster:
                s+=point[i]
            up_center.append(s/len(cluster))
        centers.append(up_center)
    return centers    

#returns a tuple of clusters, clusts_ind= list of indexes of the clusters each point was assigned to
#given the current centroids and the points, sorts the points
#to their closest centroid 
def sort_points(points, centroids):
    """
    Sorts points to nearest centroid.

    Args:
        points (list[list[float]]): Input points.
        centroids (list[list[float]]): Current centroids.

    Returns:
            clusters (list[list[list[float]]]): Points grouped by nearest centroid.
            clusts_ind (list[int]): List of indexes of the clusters each point was assigned to.
    """
    clusters = [[] for i in range(len(centroids))]
    clusts_ind = [] 
    for p_ind in range(len(points)): 
        point = points[p_ind] 
        minDist = math.inf
        centIdx = len(centroids)
        for i in range(len(centroids)):
            centroid = centroids[i]
            dist = getDistance(centroid, point)
            if dist<minDist:
                minDist = dist
                centIdx = i
        clusters[centIdx].append(point)
        clusts_ind.append(centIdx) 
    return clusters, clusts_ind 

#checks if the centroids converged enough
def e_convergence(prev_ctr, up_ctr):
     """
    Check if centroids converged.

    Args:
        prev_ctr (list[list[float]]): Previous centroids.
        up_ctr (list[list[float]]): Updated centroids.

    Returns:
        bool: True if converged, False otherwise.
    """
     for i in range(len(prev_ctr)):
        if(getDistance(prev_ctr[i],up_ctr[i]))>=EPSILON:
            return False
     return True

def final_clusters(points_arr, k):
    """
    Cluster the data points into k clusters until convergence.

    Args:
        points_arr (numpy.ndarray): Input points as NumPy array.
        k (int): Number of clusters.

    Returns:
        numpy.ndarray: Cluster indices each point was assigned to.
    """
    points = points_arr.tolist()
    centroids = [points[i] for i in range(k)]
    for i in range(300): #default iter is 300
            clusters, clusters_indexs = sort_points(points, centroids)
            new_cents = update_center(clusters)
            if e_convergence(centroids, new_cents):
                break
            centroids = new_cents
    final_clusters_indexs = np.array(clusters_indexs)
    return final_clusters_indexs            
