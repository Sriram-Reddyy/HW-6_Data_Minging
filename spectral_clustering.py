"""
Work with Spectral clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle

def spectral(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[
    NDArray[np.int32] | None, float | None, float | None, NDArray[np.floating] | None
]:
    # Function to calculate the Euclidean distance matrix
    def calculate_distance_matrix(data):
        num_points = data.shape[0]
        distance_matrix = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(num_points):
                distance_matrix[i, j] = np.linalg.norm(data[i] - data[j])
        return distance_matrix

    # Convert distances to similarities using a Gaussian kernel
    def distance_to_similarity(distance_matrix, sigma):
        similarity_matrix = np.exp(-distance_matrix**2 / (sigma**2))
        np.fill_diagonal(similarity_matrix, 0) 
        return similarity_matrix

    distance_matrix = calculate_distance_matrix(data)
    similarity_matrix = distance_to_similarity(distance_matrix, sigma=params_dict['sigma'])
    diagonal_matrix=np.zeros_like(similarity_matrix)
    diag_elements=np.sum(similarity_matrix,axis=1)

    for i in range(diagonal_matrix.shape[0]):
      diagonal_matrix[i][i]=diag_elements[i]

    sigma = params_dict['sigma']
    k = params_dict['k']

    L = diagonal_matrix - similarity_matrix
    values,vectors=np.linalg.eigh(L)
    eigen_val=np.diag(values)
    v1=vectors[:,:params_dict['k']]
    normalized_vectors=v1/np.linalg.norm(v1,axis=1,keepdims=True)

    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)[:k]  # Sorting eigenvalues
    V = eigenvectors[:, idx] 

    def initialize_centroids(data, k):
        indices = np.random.choice(data.shape[0], k, replace=False)
        return data[indices]

    def assign_clusters(data, centroids):
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def update_centroids(data, assignments, k):
        new_centroids = np.array([data[assignments == i].mean(axis=0) if np.any(assignments == i) else data[np.random.choice(data.shape[0])] for i in range(k)])
        return new_centroids

    def compute_sse(data, centroids, assignments):
        sse = 0
        for k in range(centroids.shape[0]):
            cluster_data = data[assignments == k]
            sse += np.sum((cluster_data - centroids[k])**2)
        return sse

    def k_means(data, k, max_iters=300):
        centroids = initialize_centroids(data, k)
        previous_sse = None

        for iteration in range(max_iters):
            assignments = assign_clusters(data, centroids)
            new_centroids = update_centroids(data, assignments, k)
            sse = compute_sse(data, new_centroids, assignments)

            if sse == 0:
                if previous_sse is None or previous_sse == 0:  # Check if re-initialization is needed
                    print("SSE reached zero, re-initializing centroids...")
                    centroids = initialize_centroids(data, k)
                    continue
                else:
                    break

            if np.allclose(centroids, new_centroids, atol=1e-6):
                break

            centroids = new_centroids
            previous_sse = sse

        return centroids, assignments, sse

    centroids, assignments,sse = k_means(normalized_vectors, params_dict['k'])

    def adjusted_rand_index(labels_true, labels_pred):
        classes = np.unique(labels_true)
        clusters = np.unique(labels_pred)

        contingency_table = np.zeros((classes.size, clusters.size), dtype=int)
        for class_idx, class_label in enumerate(classes):
            for cluster_idx, cluster_label in enumerate(clusters):
                contingency_table[class_idx, cluster_idx] = np.sum((labels_true == class_label) & (labels_pred == cluster_label))

        sum_over_rows = np.sum(contingency_table, axis=1)
        sum_over_cols = np.sum(contingency_table, axis=0)

        n_combinations = sum([n_ij * (n_ij - 1) / 2 for n_ij in contingency_table.flatten()])
        sum_over_rows_comb = sum([n_ij * (n_ij - 1) / 2 for n_ij in sum_over_rows])
        sum_over_cols_comb = sum([n_ij * (n_ij - 1) / 2 for n_ij in sum_over_cols])

        n = labels_true.size
        total_combinations = n * (n - 1) / 2
        expected_index = sum_over_rows_comb * sum_over_cols_comb / total_combinations
        max_index = (sum_over_rows_comb + sum_over_cols_comb) / 2
        denominator = (max_index - expected_index)
        
        if denominator == 0:
            return 1 if n_combinations == expected_index else 0

        ari = (n_combinations - expected_index) / denominator
        return ari

    labels_true = labels 
    labels_pred = assignments

    ari = adjusted_rand_index(labels_true, labels_pred)

    computed_labels: NDArray[np.int32] | None = assignments
    SSE: float | None = sse
    ARI: float | None = ari
    eigenvalues: NDArray[np.floating] | None = values
    return computed_labels, SSE, ARI, eigenvalues

def spectral_clustering():

    answers = {}
    data=np.load("question1_cluster_data.npy")
    true_labels=np.load("question1_cluster_labels.npy")

    answers["spectral_function"] = spectral

    groups = {}
    sse_final=[]
    preds_final=[]
    ari_final=[]
    eigen_final=[]

    sigma2 = np.linspace(0.1,10,10);

    for i in range(5):
      datav=data[i*2000:(i+1)*2000]
      true_labelsv=true_labels[i*2000:(i+1)*2000]
    # for i in np.arange(1,10,0.1):
      params_dict={'k':5,'sigma':0.1}
      preds,sse_hyp,ari_hyp,eigen_val=spectral(datav,true_labelsv,params_dict)
      print(f"ARI values are {ari_hyp}")
      sse_final.append(sse_hyp)
      ari_final.append(ari_hyp)
      preds_final.append(preds)
      eigen_final.append(eigen_val)
      for i in range(len(sigma2)):
        groups[i]={'sigma':sigma2[i],'ARI':ari_hyp,"SSE":sse_hyp}
        # groups[i]['SSE']=sse_hyp
        # groups[i]['ARI']=ari_hyp
      else:
        pass

    sse_numpy=np.array(sse_final)
    ari_numpy=np.array(ari_final)
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[0]['SSE']

    least_sse_index=np.argmin(sse_numpy)
    highest_ari_index=np.argmax(ari_numpy)
    lowest_ari_index=np.argmin(ari_numpy)
    #print(least_sse_index,highest_ari_index)
    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.
    # for i in groups:
    #   if groups[i]['SSE']>
    # # Plot is the return value of a call to plt.scatter()
    #print(1000*highest_ari_index,(highest_ari_index+1)*1000)

    # [i*1000:(i+1)*5000-1]
    plot_ARI=plt.scatter(data[2000*highest_ari_index:(highest_ari_index+1)*2000, 0], data[2000*highest_ari_index:(highest_ari_index+1)*2000, 1], c=preds_final[highest_ari_index], cmap='viridis', marker='.')
    # plt.scatter(true_labelsv[:, 0], true_labelsv[:, 1], c=datav, cmap='viridis', marker='.')
    plt.title('Largest ARI')
    plt.xlabel(f'Feature 1 for Dataset{i+1}')
    plt.ylabel(f'Feature 2 for Dataset{i+1}')
    # plt.colorbar()
    plt.grid(True)
    plt.savefig('SpectralClustering_LargestARI.png')
    # plt.show()


    #print(1000*least_sse_index,(least_sse_index+1)*1000-1)
    
    # [i*1000:(i+1)*1000-1]
    
    plot_SSE=plt.scatter(data[2000*least_sse_index:(least_sse_index+1)*2000, 0], data[2000*least_sse_index:(least_sse_index+1)*2000, 1], c=preds_final[least_sse_index], cmap='viridis', marker='.')
    # plt.scatter(true_labelsv[:, 0], true_labelsv[:, 1], c=datav, cmap='viridis', marker='.')
    plt.title('Least SSE')
    plt.xlabel(f'Feature 1 for Dataset{i+1}')
    plt.ylabel(f'Feature 2 for Dataset{i+1}')
    plt.grid(True)
    plt.savefig('SpectralClusteringOutputLeastSSE.png')
    # plt.colorbar()
    # plt.show()
    # plot_ARI = plt.scatter([1,2,3], [4,5,6])
    # plot_SSE = plt.scatter([1,2,3], [4,5,6])
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # # Plot of the eigenvalues (smallest to largest) as a line plot.
    # # Use the plt.plot() function. Make sure to include a title, axis labels, and a grid.

    value_to_plot_eva=[]
    for i in range(len(eigen_final)):
      for val in eigen_final[i]:
        value_to_plot_eva.append(val)


    plt.title('Eigen Values Sorted')
    plot_eig=plt.plot(sorted(value_to_plot_eva))
    plt.plot(sorted(value_to_plot_eva))
    plt.xlabel(f'Eigen Values Sorted in Ascending')
    plt.grid(True)
    plt.savefig('SpectralClustering_Eigenvalues.png')
    

    answers['eigenvalue plot']=plot_eig
    plt.close()

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

    # A single float
    answers["std_ARIs"] = float(np.std(ari_numpy))

    # A single float
    answers["mean_SSEs"] = float(np.mean(sse_numpy))

    # A single float
    answers["std_SSEs"] = float(np.std(sse_numpy))

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = spectral_clustering()
    with open("spectral_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
