import numpy as np
import torch
from tqdm import tqdm


def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def kmeans(
        X_FULL,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
        trunk_size=2560000, # this actually not limited here because we have implemented pairwise_distance_euclidean_mem_efficient_batched
        device=torch.device('cpu')
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    print(f'running k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'euclidean_mem_efficient':
        pairwise_distance_function = pairwise_distance_euclidean_mem_efficient_batched
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    initial_state = None

    # convert to float
    X_FULL = X_FULL.float()
    dataset_size=len(X_FULL)

    iteration = 0
    tqdm_meter = tqdm(desc='[running kmeans]')
    initial_state_list = []

    # Three epochs
    for epoch in range(25):
        print("Epoch", epoch)
        X = None
        start_index = 0
        rand_indices = torch.randperm(X_FULL.size(0))
        X_FULL = X_FULL[rand_indices]
        # For each epochs
        while True:
            # A subset of data
            if X is None:
                X = X_FULL[start_index:start_index + trunk_size].to(device)
                print("Process data from {} to {}".format(start_index, start_index + trunk_size))

            # This initial_state is iteratively updated by the dataset
            if initial_state is None:
                initial_state = initialize(X, num_clusters)

            dis = pairwise_distance_function(X, initial_state, device=device)

            choice_cluster = torch.argmin(dis, dim=1)

            initial_state_pre = initial_state.clone()

            non_matched_centroid = 0
            for index in range(num_clusters):
                selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

                selected = torch.index_select(X, 0, selected)
                if len(selected) == 0:
                    initial_state[index] = initial_state_pre[index]
                    non_matched_centroid += 1
                else:
                    initial_state[index] = selected.mean(dim=0)

            center_shift = torch.sum(
                torch.sqrt(
                    torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
                ))

            # increment iteration
            iteration = iteration + 1

            # update tqdm meter
            tqdm_meter.set_postfix(
                iteration=f'{iteration}',
                center_shift=f'{center_shift ** 2:0.6f}',
                tol=f'{tol:0.6f}',
                non_matched_centroid=f'{non_matched_centroid:0.1f}'
            )
            tqdm_meter.update()
            
            if center_shift ** 2 < tol or iteration > 20000:
                start_index += trunk_size
                X = None
                initial_state_list.append(initial_state.clone().cpu())

                if start_index + trunk_size < dataset_size:
                    continue
                else:
                    # Full data is processed
                    break

    return choice_cluster.cpu(), initial_state.cpu()


def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cpu')
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) cluster ids
    """
    print(f'predicting on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'euclidean_mem_efficient':
        pairwise_distance_function = pairwise_distance_euclidean_mem_efficient
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()

def pairwise_distance_euclidean_mem_efficient_batched(data1, data2, device=torch.device('cpu'), batch_size=256):
    """
    Compute pairwise Euclidean distance in a memory-efficient way, suitable for large datasets.
    
    Args:
    - data1 (Tensor): Tensor of shape [N, D], where N is the number of vectors and D is the dimensionality.
    - data2 (Tensor): Tensor of shape [M, D], where M is the number of vectors and D is the dimensionality.
    - device (torch.device): The device (CPU/GPU) on which to perform the computation.
    - batch_size (int): The size of each batch to be processed. Adjust based on your GPU's memory capacity.
    
    Returns:
    - distances (Tensor): A tensor containing the pairwise distances between each pair of vectors in data1 and data2.
    """
    # Transfer data2 to GPU and compute its squared norm
    data2 = data2.to(device)
    norm2 = data2.pow(2).sum(dim=1, keepdim=True)

    # Initialize a tensor to hold the computed distances
    distances = torch.zeros(data1.shape[0], data2.shape[0], device=device)
    
    # Process data1 in batches to save memory
    for i in range(0, data1.shape[0], batch_size):
        # Compute the end index of the current batch
        end = min(i + batch_size, data1.shape[0])
        
        # Transfer the current batch to GPU and compute its squared norm
        batch_data1 = data1[i:end].to(device)
        norm1 = batch_data1.pow(2).sum(dim=1, keepdim=True)
        
        # Compute dot products between the current batch and data2
        dot_product = torch.mm(batch_data1, data2.transpose(0, 1))
        
        # Compute distances for the current batch and store them
        distances[i:end, :] = norm1 + norm2.transpose(0, 1) - 2 * dot_product
    
    # Optionally, move the distances matrix back to CPU if further processing is needed there
    # distances = distances.cpu()
    
    return distances


def pairwise_distance_euclidean_mem_efficient(data1, data2, device=torch.device('cpu')):
    # data1 shape: [1598000, 768]
    # data2 shape: [4096, 768]
    data1, data2 = data1.to(device), data2.to(device)

    # Compute squared norms of each row in data1 and data2
    norm1 = data1.pow(2).sum(dim=1, keepdim=True)
    norm2 = data2.pow(2).sum(dim=1, keepdim=True)

    # Compute dot products
    # Transpose data2 to make its shape compatible for dot product
    dot_product = torch.mm(data1, data2.transpose(0, 1))

    return norm1 + norm2.transpose(0, 1) - 2 * dot_product

def pairwise_distance(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)
    
    # OOM here:
    # ipdb> A.size()
    # torch.Size([199000, 1, 768])
    # ipdb> B.size()
    # torch.Size([1, 1024, 768])
    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis

