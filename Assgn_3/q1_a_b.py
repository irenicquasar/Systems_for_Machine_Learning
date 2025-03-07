import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import os
import copy
from tqdm import tqdm

# Have used code of LeNET-5 from the previous assignment 
# RBF Layer
class RBFLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centers = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        x_expanded = x.unsqueeze(1).expand(-1, self.out_features, -1)
        centers_expanded = self.centers.unsqueeze(0).expand(x.size(0), -1, -1)
        distances = torch.sum((x_expanded - centers_expanded) ** 2, dim=2)
        return distances

# MAPLoss
class MAPLoss(nn.Module):
    def __init__(self, j=1.0):
        super().__init__()
        self.j = j
    
    def forward(self, outputs, targets):
        batch_size = outputs.size(0)
        correct_class_distances = outputs[torch.arange(batch_size), targets]
        exp_neg_distances = torch.exp(-outputs)
        j_tensor = torch.tensor(self.j, dtype=outputs.dtype, device=outputs.device)
        exp_neg_j = torch.exp(-j_tensor)
        sum_exp_terms = exp_neg_distances.sum(dim=1) + exp_neg_j
        log_sum_exp_term = torch.log(sum_exp_terms)
        loss = (correct_class_distances + log_sum_exp_term).mean()
        return loss

# LeNet-5 Architecture
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.rbf = RBFLayer(84, 10)

    def forward(self, x):
        x = self.pool1(torch.tanh(self.conv1(x)))
        x = self.pool2(torch.tanh(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.rbf(x)
        return x

# All-Reduce 
def all_reduce(gradients_list, world_size):
    reduced_gradients = {}
    for name in gradients_list[0].keys():
        reduced_gradients[name] = torch.zeros_like(gradients_list[0][name])
        for rank in range(world_size):
            reduced_gradients[name] += gradients_list[rank][name]
        reduced_gradients[name] /= world_size
    return [reduced_gradients for _ in range(world_size)]

# Ring all reduce 1D ( I have referred to the medium article  :https://medium.com/towards-data-science/visual-intuition-on-ring-allreduce-for-distributed-deep-learning-d1f34b4911da )
# for hte exact concept of ring all reduce and implemented the same in the below code 

def ring_allreduce_1d(param_tensors):
    """
    Perform a two-phase ring allreduce on a list of 1D tensors.
    
    Args:
      param_tensors (List[Tensor]): Each worker's 1D tensor.
      
    Returns:
      List[Tensor]: A list of fully reduced (and identical) tensors, one per worker.
    """
    p = len(param_tensors)
    n = param_tensors[0].numel()
    chunk_size = n // p  

    # Split each worker's tensor into p chunks
    chunk_buffers = []
    for r in range(p):
        chunks = list(param_tensors[r].split(chunk_size))
        chunk_buffers.append(chunks)  # chunk_buffers[r] is a list of p chunks

    # Phase 1: Share-Reduce Phase 
    # For s = 0 to p-2 (p-1 steps)
    for s in range(p - 1):
        # Each worker i sends chunk index (i - s) mod p to worker (i+1) mod p.
        send_chunks = [None] * p
        recv_chunks = [None] * p
        
        # Identify chunks to send for each worker.
        for i in range(p):
            send_idx = (i - s) % p
            send_chunks[i] = chunk_buffers[i][send_idx]

        # Simulate ring communication:
        for i in range(p):
            # Worker i receives from worker (i-1) mod p.
            prev = (i - 1) % p
            recv_chunks[i] = send_chunks[prev]
        
        # Each worker adds the received chunk to its corresponding chunk.
        for i in range(p):
            recv_idx = (i - s - 1) % p
            chunk_buffers[i][recv_idx] = chunk_buffers[i][recv_idx] + recv_chunks[i]

    # Phase 2: Share-Only Phase 
    # Now each worker has one fully reduced chunk.
    # We circulate these chunks so that every worker obtains every chunk.
    for s in range(p - 1):
        send_chunks = [None] * p
        recv_chunks = [None] * p
        
        for i in range(p):
            send_idx = (i - s) % p
            send_chunks[i] = chunk_buffers[i][send_idx]
        
        for i in range(p):
            prev = (i - 1) % p
            recv_chunks[i] = send_chunks[prev]
        
        # In this phase, we just copy (no addition) the received chunk to the appropriate slot.
        for i in range(p):
            recv_idx = (i - s - 1) % p
            chunk_buffers[i][recv_idx] = recv_chunks[i]

    # Reassemble the full tensor for each worker by concatenating its chunks.
    final_tensors = [torch.cat(chunks, dim=0) for chunks in chunk_buffers]
    return final_tensors


def ring_all_reduce(gradients_list, world_size):
    """
    Perform ring allreduce across all gradient dictionaries.
    
    Args:
      gradients_list (List[Dict[str, Tensor]]): List of gradient dictionaries (one per worker).
      world_size (int): Number of workers.
      
    Returns:
      List[Dict[str, Tensor]]: Each element is a dictionary with reduced (averaged) gradients.
    """
    reduced_gradients = [{} for _ in range(world_size)]
    
    for name in gradients_list[0].keys():
        # Flatten each worker's gradient tensor to 1D.
        worker_tensors = [gradients_list[r][name].flatten() for r in range(world_size)]
        
        # Use ring allreduce to sum the gradients.
        reduced_tensors = ring_allreduce_1d(worker_tensors)
        
        # Divide by world_size to get the average.
        reduced_tensors = [rt / world_size for rt in reduced_tensors]
        
        # Reshape the 1D tensor back to the original shape.
        original_shape = gradients_list[0][name].shape
        for r in range(world_size):
            reduced_gradients[r][name] = reduced_tensors[r].view(original_shape)
            
    return reduced_gradients


# Worker function for training
def worker(rank, world_size, model, criterion, optimizer, train_indices, test_dataset, 
           epochs, sync_method, gradients_queue, results_queue, sync_barrier):
    train_subset = Subset(test_dataset, train_indices[rank])
    train_loader = DataLoader(train_subset, batch_size=10, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            gradients = {name: param.grad.clone() for name, param in model.named_parameters() if param.requires_grad and param.grad is not None}
            gradients_queue.put((rank, gradients))
            sync_barrier.wait()
            if rank == 0:
                all_gradients = [None] * world_size
                for _ in range(world_size):
                    worker_rank, worker_gradients = gradients_queue.get()
                    all_gradients[worker_rank] = worker_gradients
                if sync_method == 'all-reduce':
                    reduced_gradients = all_reduce(all_gradients, world_size)
                else:
                    reduced_gradients = ring_all_reduce(all_gradients, world_size)
                for r in range(world_size):
                    gradients_queue.put((r, reduced_gradients[r]))
            sync_barrier.wait()
            while True:
                worker_rank, reduced_gradient = gradients_queue.get()
                if worker_rank == rank:
                    for name, param in model.named_parameters():
                        if param.requires_grad and name in reduced_gradient:
                            param.grad = reduced_gradient[name]
                    break
                else:
                    gradients_queue.put((worker_rank, reduced_gradient))
            optimizer.step()
            sync_barrier.wait()
    
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=100)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.min(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    results_queue.put((rank, accuracy))

# Main function
def main(sync_method='all-reduce'):
    mp.set_start_method('spawn', force=True)
    world_size = 4
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    all_indices = np.random.choice(len(test_dataset), 100, replace=False)
    train_indices = np.array_split(all_indices, world_size)
    model = LeNet5()
    model.share_memory()
    criterion = MAPLoss(j=1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    gradients_queue = mp.Queue()
    results_queue = mp.Queue()
    sync_barrier = mp.Barrier(world_size)
    processes = []
    start_time = time.time()
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, model, criterion, optimizer, train_indices, test_dataset, 100, sync_method, gradients_queue, results_queue, sync_barrier))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    training_time = time.time() - start_time
    accuracies = []
    while not results_queue.empty():
        rank, accuracy = results_queue.get()
        accuracies.append((rank, accuracy))
    avg_accuracy = sum(acc for _, acc in accuracies) / len(accuracies)
    print(f"Sync Method: {sync_method}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Average Accuracy: {avg_accuracy:.2f}%")
    return training_time, avg_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Parallel LeNet-5 Training")
    parser.add_argument("sync_method", choices=["all-reduce", "ring-all-reduce"], help="Synchronization method for gradient updates")
    args = parser.parse_args()
    main(args.sync_method)