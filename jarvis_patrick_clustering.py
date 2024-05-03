"""
Work with Jarvis-Patrick clustering.
Do not use global variables!
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################

def jarvis_patrick(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[NDArray[np.int32] | None, float | None, float | None]:
    """
    Implementation of the Jarvis-Patrick algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'k', 'smin', There could be others.
    - params_dict['k']: the number of nearest neighbors to consider. This determines the size of the neighborhood used to assess the similarity between datapoints.
    Choose values in the range 3 to 8
    - params_dict['smin']:  the minimum number of shared neighbors to consider two points in the same cluster.
       Choose values in the range 4 to 10.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index

    Notes:
    - the nearest neighbors can be bidirectional or unidirectional
    - Bidirectional: if point A is a nearest neighbor of point B, then point B is a nearest neighbor of point A).
    - Unidirectional: if point A is a nearest neighbor of point B, then point B is not necessarily a nearest neighbor of point A).
    - In this project, only consider unidirectional nearest neighboars for simplicity.
    - The metric  used to compute the the k-nearest neighberhood of all points is the Euclidean metric
    """
    true_labels=labels
    def create_shared_neighbor_matrix(data, k, t):
        distance_matrix = squareform(pdist(data, 'euclidean'))
        neighbors = np.argsort(distance_matrix, axis=1)[:, 1:k+1]
        n = len(data)
        adjacency_matrix = np.zeros((n, n), dtype=bool)

        for i in range(n):
            for j in range(i + 1, n):
                shared_neighbors = len(set(neighbors[i]).intersection(neighbors[j]))
                if shared_neighbors >= t:
                    adjacency_matrix[i, j] = True
                    adjacency_matrix[j, i] = True

        return adjacency_matrix

    def calculate_sse(data, labels, cluster_centers):
        sse = 0
        for k in range(len(cluster_centers)):
            cluster_data = data[labels == k]
            sse += np.sum((cluster_data - cluster_centers[k])**2)
        return sse
    def adjusted_rand_index(labels_true, labels_pred):
        # Find the unique classes and clusters
        classes = np.unique(labels_true)
        clusters = np.unique(labels_pred)

        # Create the contingency table
        contingency_table = np.zeros((classes.size, clusters.size), dtype=int)
        for class_idx, class_label in enumerate(classes):
            for cluster_idx, cluster_label in enumerate(clusters):
                contingency_table[class_idx, cluster_idx] = np.sum((labels_true == class_label) & (labels_pred == cluster_label))

        # Compute the sum over the rows and columns
        sum_over_rows = np.sum(contingency_table, axis=1)
        sum_over_cols = np.sum(contingency_table, axis=0)

        # Compute the number of combinations of two
        n_combinations = sum([n_ij * (n_ij - 1) / 2 for n_ij in contingency_table.flatten()])
        sum_over_rows_comb = sum([n_ij * (n_ij - 1) / 2 for n_ij in sum_over_rows])
        sum_over_cols_comb = sum([n_ij * (n_ij - 1) / 2 for n_ij in sum_over_cols])

        # Compute terms for the adjusted Rand index
        n = labels_true.size
        total_combinations = n * (n - 1) / 2
        expected_index = sum_over_rows_comb * sum_over_cols_comb / total_combinations
        max_index = (sum_over_rows_comb + sum_over_cols_comb) / 2
        denominator = (max_index - expected_index)

        # Handle the special case when the denominator is 0
        if denominator == 0:
            return 1 if n_combinations == expected_index else 0

        ari = (n_combinations - expected_index) / denominator

        return ari
    def dbscan_custom(matrix, data, minPts):
        n = matrix.shape[0]
        labels = -np.ones(n)
        cluster_id = 0
        cluster_centers = []

        for i in range(n):
            if labels[i] != -1:
                continue
            neighbors = np.where(matrix[i])[0]
            if len(neighbors) < minPts:
                labels[i] = -2
                continue
            labels[i] = cluster_id
            seed_set = set(neighbors)

            # Start a new cluster and determine the center
            cluster_points = [data[i]]

            while seed_set:
                current_point = seed_set.pop()
                cluster_points.append(data[current_point])
                if labels[current_point] == -2:
                    labels[current_point] = cluster_id
                if labels[current_point] != -1:
                    continue
                labels[current_point] = cluster_id
                current_neighbors = np.where(matrix[current_point])[0]
                if len(current_neighbors) >= minPts:
                    seed_set.update(current_neighbors)

            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers.append(cluster_center)
            cluster_id += 1

        return labels, np.array(cluster_centers)

    # Example usage:
    # data = np.random.rand(50, 2)  # Random 2D points
    adjacency_matrix = create_shared_neighbor_matrix(data, k=params_dict['k'], t=2)
    labels, cluster_centers = dbscan_custom(adjacency_matrix, data, minPts=params_dict['smin'])
    sse = calculate_sse(data, labels, cluster_centers)

    # true_labels = np.random.randint(0, 2, size=50)  # Placeholder for true labels
    ari = adjusted_rand_index(true_labels, labels)
#     ari_orig=adjusted_rand_score(true_labels,labels)
    computed_labels: NDArray[np.int32] | None = labels
    SSE: float | None = sse
    ARI: float | None = ari

    return computed_labels, SSE, ARI


def jarvis_patrick_clustering():
    """
    Performs Jarvis-Patrick clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """


    answers = {}
    data=np.load("question1_cluster_data.npy")
    true_labels=np.load("question1_cluster_labels.npy")
    # Return your `spectral` function
    answers["jarvis_patrick_function"] = jarvis_patrick

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Spectral clustering.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}

    # For the spectral method, perform your calculations with 5 clusters.
    # In this cas,e there is only a single parameter, σ.

# Hyperparameter Tuning Code for best K and smin
    # sse=[]
    # ari=[]
    # predictions=[]
    # k_values=[3,4,5,6,7,8]
    # for counter,i in enumerate(k_values):
    #   datav=data[:1000]
    #   true_labelsv=true_labels[:1000]
    # # for i in np.arange(4,11,1):
        #   params_dict={'k':5,'smin':i}
        #   preds,sse_hyp,ari_hyp,eigen_val=jarvis_patrick(datav,true_labelsv,params_dict)
        #   sse.append(sse_hyp)
        #   ari.append(ari_hyp)
        #   predictions.append(preds)
    #   if counter not in groups:
    #     # groups[counter]={'sigma':0.1,'ARI':ari_hyp,"SSE":sse_hyp}
    #     pass
    #     # groups[i]['SSE']=sse_hyp
    #     # groups[i]['ARI']=ari_hyp
    #   else:
    #     pass
    # sse_numpy=np.array(sse)
    # ari_numpy=np.array(ari)

    #  k = 3 - 8
    #  smin = 4 - 10

    sse_final=[]
    preds_final=[]
    ari_final=[]
    eigen_final=[]
    for i in range(5):
      datav=data[i*1000:(i+1)*1000]
      true_labelsv=true_labels[i*1000:(i+1)*1000]
    # for i in np.arange(1,10,0.1):
      params_dict={'k':5,'smin':5}
      preds,sse_hyp,ari_hyp,=jarvis_patrick(datav,true_labelsv,params_dict)
      sse_final.append(sse_hyp)
      ari_final.append(ari_hyp)
      preds_final.append(preds)
      if i not in groups:
        groups[i]={'k':5,'smin':5,'ARI':ari_hyp,"SSE":sse_hyp}
        # groups[i]['SSE']=sse_hyp
        # groups[i]['ARI']=ari_hyp
      else:
        pass

    sse_numpy=np.array(sse_final)
    ari_numpy=np.array(ari_final)
    # print(groups)
    # plt.plot(ari)
    # print(preds,sse,ari,eigen_val)
    # print(sse,ari)
    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}
    # for
    # print(groups)
    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[0]['SSE']

    # Identify the cluster with the lowest value of ARI. This implies
    # that you set the cluster number to 5 when applying the spectral
    # algorithm.

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.
    least_sse_index=np.argmin(sse_numpy)
    highest_ari_index=np.argmax(ari_numpy)
    lowest_ari_index=np.argmin(ari_numpy)
    # print(least_sse_index,highest_ari_index)
    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.
    # for i in groups:
    #   if groups[i]['SSE']>
    # # Plot is the return value of a call to plt.scatter()
    # print(1000*highest_ari_index,(highest_ari_index+1)*1000)
    # plt.figure(figsize=(8, 6))
    # [i*1000:(i+1)*1000-1]
    plot_ARI=plt.scatter(data[1000*highest_ari_index:(highest_ari_index+1)*1000, 0], data[1000*highest_ari_index:(highest_ari_index+1)*1000, 1], c=preds_final[highest_ari_index], cmap='viridis', marker='.')
    # plt.scatter(true_labelsv[:, 0], true_labelsv[:, 1], c=datav, cmap='viridis', marker='.')
    plt.title('Largest ARI')
    plt.xlabel(f'Feature 1 for Dataset{i+1}')
    plt.ylabel(f'Feature 2 for Dataset{i+1}')
    # plt.colorbar()
    plt.grid(True)
    plt.savefig("LargestARI_JP.png")
    # plt.show()


    # print(1000*least_sse_index,(least_sse_index+1)*1000-1)
    # plt.figure(figsize=(8, 6))
    # [i*1000:(i+1)*1000-1]

    plot_SSE=plt.scatter(data[1000*least_sse_index:(least_sse_index+1)*1000, 0], data[1000*least_sse_index:(least_sse_index+1)*1000, 1], c=preds_final[least_sse_index], cmap='viridis', marker='.')
    # plt.scatter(true_labelsv[:, 0], true_labelsv[:, 1], c=datav, cmap='viridis', marker='.')
    plt.title('Least SSE')
    plt.xlabel(f'Feature 1 for Dataset{i+1}')
    plt.ylabel(f'Feature 2 for Dataset{i+1}')
    plt.grid(True)
    plt.savefig("SmallestSSE_JP.png")
    plt.close()
    # plt.colorbar()
    # plt.show()
    # plot_ARI = plt.scatter([1,2,3], [4,5,6])
    # plot_SSE = plt.scatter([1,2,3], [4,5,6])
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # # Plot of the eigenvalues (smallest to largest) as a line plot.
    # # Use the plt.plot() function. Make sure to include a title, axis labels, and a grid.


    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.
    ARI_sum=[]
    SSE_sum=[]
    for i in groups:
      if 'ARI' in groups[i]:
        ARI_sum.append(groups[i]['ARI'])
        SSE_sum.append(groups[i]['SSE'])

    # A single float
    answers["mean_ARIs"] = float(np.mean(ari_numpy))
    # print(type(float(np.mean(np.array(ARI_sum)))))

    # A single float
    answers["std_ARIs"] = float(np.std(ari_numpy))
    # print(type(np.std(np.array(ARI_sum))))

    # A single float
    answers["mean_SSEs"] = float(np.mean(sse_numpy))

    # A single float
    answers["std_SSEs"] = float(np.std(sse_numpy))

    return answers

# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = jarvis_patrick_clustering()
    with open("jarvis_patrick_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
