def k_means_assignment(points: list, centroids: list) -> list:
    """
    Returns the nearest-centroid index for every point.
    """
    centroid_index = []
    for pt in points : 
        centroid_distances = {}
        for centroid in centroids :
            dist = sqrd_euclid_dist(pt, centroid)
            centroid_distances[tuple(centroid)] = dist 

        smallest_centroid = min(centroid_distances, key=centroid_distances.get)
        
        centroid_index.append(centroids.index(list(smallest_centroid)))

    return centroid_index

            


def sqrd_euclid_dist(x: list, y: list) -> float : 
    return float(sum([(a-b)**2 for a,b in zip(x,y)]))