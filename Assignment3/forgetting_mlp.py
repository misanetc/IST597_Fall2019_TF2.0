# -*- coding: utf-8 -*-
"""
Author:-aam35
Analyzing Forgetting in neural networks
Catastrophic Forgetting Experiment with Permuted MNIST
"""

import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import csv
import random

# Set random seeds for reproducibility
SEED = 1234
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Load MNIST data
print("Loading MNIST data...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Flatten and normalize
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = tf.one_hot(y_train, depth=10).numpy()
y_test = tf.one_hot(y_test, depth=10).numpy()

# Create a simple data class to mimic the old interface
class DataSet:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

class Data:
    def __init__(self, x_train, y_train, x_test, y_test):
        # Use first 55000 for training, rest for validation
        self.train = DataSet(x_train[:55000], y_train[:55000])
        self.validation = DataSet(x_train[55000:], y_train[55000:])
        self.test = DataSet(x_test, y_test)

data = Data(x_train, y_train, x_test, y_test)

# Experiment parameters
num_tasks_to_run = 10
num_epochs_first_task = 50
num_epochs_per_task = 20
minibatch_size = 32
learning_rate = 0.001

# Model parameters
size_input = 784  # MNIST data input (img shape: 28*28)
size_hidden = 256
size_output = 10  # MNIST total classes (0-9 digits)

# Generate task permutations
print("Generating task permutations...")
task_permutation = []
for task in range(num_tasks_to_run):
    np.random.seed(SEED + task)
    task_permutation.append(np.random.permutation(784))

# Define MLP class with variable depth
class MLP(object):
    def __init__(self, size_input, size_hidden, size_output, depth=2, 
                 dropout_rate=0.0, device=None):
        """
        size_input: int, size of input layer
        size_hidden: int, size of hidden layer
        size_output: int, size of output layer
        depth: int, number of hidden layers (2, 3, or 4)
        dropout_rate: float, dropout rate (0.0 to 0.5)
        device: str or None, either 'cpu' or 'gpu' or None
        """
        self.size_input = size_input
        self.size_hidden = size_hidden
        self.size_output = size_output
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.device = device
        
        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        
        # Input to first hidden layer
        self.W1 = tf.Variable(tf.random.normal([size_input, size_hidden], 
                                              stddev=0.1), name="W_1")
        self.b1 = tf.Variable(tf.zeros([1, size_hidden]), name="b_1")
        self.weights.append(self.W1)
        self.biases.append(self.b1)
        
        # Hidden layers
        for i in range(depth - 1):
            W = tf.Variable(tf.random.normal([size_hidden, size_hidden], 
                                             stddev=0.1), name=f"W_{i+2}")
            b = tf.Variable(tf.zeros([1, size_hidden]), name=f"b_{i+2}")
            self.weights.append(W)
            self.biases.append(b)
        
        # Last hidden to output layer
        self.W_out = tf.Variable(tf.random.normal([size_hidden, size_output], 
                                                  stddev=0.1), name="W_out")
        self.b_out = tf.Variable(tf.random.normal([1, size_output]), name="b_out")
        self.weights.append(self.W_out)
        self.biases.append(self.b_out)
        
        # Define variables to be updated during backpropagation
        self.variables = self.weights + self.biases
    
    def forward(self, X):
        """
        Forward pass
        X: Tensor, inputs
        """
        if self.device is not None:
            with tf.device('gpu:0' if self.device == 'gpu' else 'cpu'):
                self.y = self.compute_output(X)
        else:
            self.y = self.compute_output(X)
        return self.y
    
    def compute_output(self, X):
        """
        Custom method to obtain output tensor during forward pass
        """
        X_tf = tf.cast(X, dtype=tf.float32)
        
        # Forward through first hidden layer
        h = tf.nn.relu(tf.matmul(X_tf, self.W1) + self.b1)
        
        # Apply dropout (note: in TF2, dropout always applies the rate regardless of training mode)
        if self.dropout_rate > 0.0:
            h = tf.nn.dropout(h, rate=self.dropout_rate)
        
        # Forward through additional hidden layers
        for i in range(self.depth - 1):
            h = tf.nn.relu(tf.matmul(h, self.weights[i+1]) + self.biases[i+1])
            if self.dropout_rate > 0.0:
                h = tf.nn.dropout(h, rate=self.dropout_rate)
        
        # Output layer
        output = tf.matmul(h, self.W_out) + self.b_out
        
        return output
    
    def loss(self, y_pred, y_true, loss_type='NLL', l1_coeff=0.0, l2_coeff=0.0):
        """
        Compute loss function
        loss_type: 'NLL', 'L1', 'L2', or 'L1+L2'
        l1_coeff: coefficient for L1 regularization
        l2_coeff: coefficient for L2 regularization
        """
        y_true_tf = tf.cast(tf.reshape(y_true, (-1, self.size_output)), 
                           dtype=tf.float32)
        y_pred_tf = tf.cast(y_pred, dtype=tf.float32)
        
        # Base loss (NLL) - using TF2 softmax_cross_entropy
        base_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=y_pred_tf, labels=y_true_tf))
        
        # Regularization terms
        reg_loss = 0.0
        if l2_coeff > 0.0:
            for W in self.weights:
                reg_loss += l2_coeff * tf.nn.l2_loss(W)
        if l1_coeff > 0.0:
            for W in self.weights:
                reg_loss += l1_coeff * tf.reduce_sum(tf.abs(W))
        
        return base_loss + reg_loss
    
    def backward(self, X_train, y_train, optimizer_type='Adam', 
                 loss_type='NLL', l1_coeff=0.0, l2_coeff=0.0):
        """
        Backward pass
        optimizer_type: 'SGD', 'Adam', or 'RMSProp'
        """
        # Set up optimizer (TF2 style)
        if optimizer_type == 'SGD':
            optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_type == 'Adam':
            optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_type == 'RMSProp':
            optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        
        with tf.GradientTape() as tape:
            predicted = self.forward(X_train)
            current_loss = self.loss(predicted, y_train, loss_type, 
                                    l1_coeff, l2_coeff)
        
        grads = tape.gradient(current_loss, self.variables)
        optimizer.apply_gradients(zip(grads, self.variables))
        
        return current_loss

def accuracy_function(y_pred, y_true):
    """Compute accuracy"""
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def apply_permutation(X, permutation):
    """Apply pixel permutation to input data"""
    return X[:, permutation]

def get_task_data(task_idx, is_train=True):
    """Get data for a specific task"""
    if is_train:
        images = data.train.images
        labels = data.train.labels
    else:
        images = data.test.images
        labels = data.test.labels
    
    # Apply permutation
    permuted_images = apply_permutation(images, task_permutation[task_idx])
    
    return permuted_images, labels

def train_model_on_task(model, task_idx, num_epochs, optimizer_type, 
                       loss_type='NLL', l1_coeff=0.0, l2_coeff=0.0, verbose=True):
    """Train model on a specific task"""
    X_train, y_train = get_task_data(task_idx, is_train=True)
    
    for epoch in range(num_epochs):
        # Create dataset with shuffling
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
            .shuffle(buffer_size=1000)\
            .batch(minibatch_size)
        
        epoch_loss = 0
        num_batches = 0
        
        for inputs, outputs in dataset:
            batch_loss = model.backward(inputs, outputs, optimizer_type, 
                                      loss_type, l1_coeff, l2_coeff)
            epoch_loss += batch_loss
            num_batches += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/num_batches:.4f}")

def test_model_on_task(model, task_idx):
    """Test model on a specific task"""
    X_test, y_test = get_task_data(task_idx, is_train=False)
    
    # Forward pass without dropout
    preds = model.compute_output(X_test)
    accuracy = accuracy_function(preds, y_test)
    
    return accuracy.numpy() * 100.0

def run_experiment(depth=2, dropout_rate=0.0, optimizer_type='Adam', 
                  loss_type='NLL', l1_coeff=0.0, l2_coeff=0.0, 
                  train_first_task=True):
    """
    Run catastrophic forgetting experiment
    
    Args:
        depth: int, number of hidden layers
        dropout_rate: float, dropout rate
        optimizer_type: str, optimizer type
        loss_type: str, loss type
        l1_coeff: float, L1 regularization coefficient
        l2_coeff: float, L2 regularization coefficient
        train_first_task: bool, whether to train on first task
    
    Returns:
        R: Task matrix
        acc: Average accuracy
        bwt: Backward transfer
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: Depth={depth}, Dropout={dropout_rate}, "
          f"Optimizer={optimizer_type}, Loss={loss_type}")
    print(f"{'='*60}")
    
    # Initialize model
    model = MLP(size_input, size_hidden, size_output, depth=depth,
               dropout_rate=dropout_rate, device='cpu')
    
    # Task matrix R[i,j] = accuracy on task j after finishing task i
    # R is triangular: we only have results for tasks we've already trained on
    R = np.zeros((num_tasks_to_run, num_tasks_to_run))
    
    # Train on first task for 50 epochs
    if train_first_task:
        print(f"\nTraining on Task 0 (50 epochs)...")
        train_model_on_task(model, 0, num_epochs_first_task, optimizer_type, 
                           loss_type, l1_coeff, l2_coeff, verbose=False)
    
    # Test on all tasks after training on task 0
    print("Testing on all tasks after training on Task 0...")
    for test_task_idx in range(num_tasks_to_run):
        acc = test_model_on_task(model, test_task_idx)
        R[0, test_task_idx] = acc
    
    print(f"Accuracies: ", end="")
    for t in range(min(5, num_tasks_to_run)):
        print(f"Task {t}: {R[0, t]:.2f}%  ", end="")
    print()
    
    # Train on subsequent tasks (20 epochs each)
    for task_idx in range(1, num_tasks_to_run):
        print(f"\nTraining on Task {task_idx} (20 epochs)...")
        train_model_on_task(model, task_idx, num_epochs_per_task, 
                          optimizer_type, loss_type, l1_coeff, l2_coeff, 
                          verbose=False)
        
        # Test on all tasks to measure forgetting
        print(f"Testing on all tasks...")
        for test_task_idx in range(num_tasks_to_run):
            acc = test_model_on_task(model, test_task_idx)
            R[task_idx, test_task_idx] = acc
        
        # Print accuracies for tasks we've trained on
        accuracies = []
        for t in range(task_idx + 1):
            accuracies.append(f"Task {t}: {R[task_idx, t]:.2f}%")
        print(f"  {'  '.join(accuracies[:5])}")  # Print first 5
    
    # Compute ACC and BWT
    T = num_tasks_to_run
    # ACC = average of R[T-1, i] over all tasks i
    acc = np.mean(R[T-1, :])
    
    # Backward transfer: R[T-1, i] - R[i, i] for i < T-1
    bwt = 0.0
    for i in range(T - 1):
        bwt += (R[T-1, i] - R[i, i])
    bwt /= (T - 1)
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Average Accuracy (ACC): {acc:.2f}%")
    print(f"  Backward Transfer (BWT): {bwt:.4f}")
    print(f"{'='*60}\n")
    
    return R, acc, bwt

def plot_results(results_dict, save_dir='results'):
    """Plot and save results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot validation accuracies (showing forgetting)
    for exp_name, (R, acc, bwt) in results_dict.items():
        plt.figure(figsize=(12, 8))
        
        # Plot accuracy for each task over time
        # Show how accuracy on each task changes as we train on new tasks
        for task_idx in range(num_tasks_to_run):
            # For task j, plot accuracy after training on tasks task_idx to num_tasks_to_run-1
            accuracies = [R[i, task_idx] for i in range(num_tasks_to_run)]
            x_values = list(range(num_tasks_to_run))
            
            plt.plot(x_values, accuracies, 
                    marker='o', label=f'Task {task_idx}')
        
        plt.xlabel('Training Task')
        plt.ylabel('Test Accuracy (%)')
        plt.title(f'Catastrophic Forgetting - {exp_name}\n'
                 f'ACC: {acc:.2f}%, BWT: {bwt:.4f}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        filename = f"{save_dir}/{exp_name.replace(' ', '_').replace('/', '_')}_forgetting.png"
        plt.savefig(filename, dpi=150)
        print(f"Saved plot: {filename}")
        plt.close()
    
    # Save results to CSV
    csv_filename = f"{save_dir}/results.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Experiment', 'ACC', 'BWT'])
        for exp_name, (R, acc, bwt) in results_dict.items():
            writer.writerow([exp_name, acc, bwt])
    
    print(f"Saved results: {csv_filename}")

def main():
    """Main function to run all experiments"""
    results = {}
    
    print("\n" + "="*60)
    print("CATASTROPHIC FORGETTING EXPERIMENTS")
    print("="*60)
    print(f"Total tasks: {num_tasks_to_run}")
    print(f"Epochs for task 0: {num_epochs_first_task}")
    print(f"Epochs per subsequent task: {num_epochs_per_task}")
    print(f"Total epochs: {num_epochs_first_task + (num_tasks_to_run-1)*num_epochs_per_task}")
    print(f"Model: {size_hidden} hidden units per layer")
    print("="*60)
    
    # Experiment 1: Baseline (Depth=2, No dropout, Adam, NLL)
    print("\n>>> Experiment 1: Baseline")
    R, acc, bwt = run_experiment(depth=2, dropout_rate=0.0, 
                                 optimizer_type='Adam', loss_type='NLL')
    results['Baseline_Depth2'] = (R, acc, bwt)
    
    # Experiment 2: Different depths
    print("\n>>> Experiment 2: Testing different depths")
    for depth in [3, 4]:
        R, acc, bwt = run_experiment(depth=depth, dropout_rate=0.0,
                                    optimizer_type='Adam', loss_type='NLL')
        results[f'Depth{depth}'] = (R, acc, bwt)
    
    # Experiment 3: Different optimizers
    print("\n>>> Experiment 3: Testing different optimizers")
    for opt in ['SGD', 'RMSProp']:
        R, acc, bwt = run_experiment(depth=2, dropout_rate=0.0,
                                     optimizer_type=opt, loss_type='NLL')
        results[f'Optimizer_{opt}'] = (R, acc, bwt)
    
    # Experiment 4: Dropout
    print("\n>>> Experiment 4: Testing dropout")
    for dropout in [0.2, 0.5]:
        R, acc, bwt = run_experiment(depth=2, dropout_rate=dropout,
                                     optimizer_type='Adam', loss_type='NLL')
        results[f'Dropout{dropout}'] = (R, acc, bwt)
    
    # Experiment 5: Different loss functions
    print("\n>>> Experiment 5: Testing different loss functions")
    for loss_type, l1, l2 in [('L1', 0.01, 0.0), ('L2', 0.0, 0.01), 
                              ('L1+L2', 0.005, 0.005)]:
        R, acc, bwt = run_experiment(depth=2, dropout_rate=0.0,
                                     optimizer_type='Adam', loss_type=loss_type,
                                     l1_coeff=l1, l2_coeff=l2)
        results[f'Loss_{loss_type}'] = (R, acc, bwt)
    
    # Plot and save results
    print("\n\nGenerating plots and saving results...")
    plot_results(results, save_dir='assignment3_outputs')
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    print(f"{'Experiment':<30} {'ACC':<12} {'BWT':<12}")
    print("-"*60)
    for exp_name, (R, acc, bwt) in results.items():
        print(f"{exp_name:<30} {acc:<12.2f} {bwt:<12.4f}")
    print("="*60)

if __name__ == '__main__':
    import sys
    
    # Allow running specific experiments via command line
    if len(sys.argv) > 1:
        if sys.argv[1] == 'baseline':
            results = {}
            R, acc, bwt = run_experiment(depth=2, dropout_rate=0.0, 
                                        optimizer_type='Adam', loss_type='NLL')
            results['Baseline'] = (R, acc, bwt)
            plot_results(results, save_dir='assignment3_outputs')
        elif sys.argv[1] == 'quick':
            # Quick test with fewer epochs
            results = {}
            R, acc, bwt = run_experiment(depth=2, dropout_rate=0.0, 
                                        optimizer_type='Adam', loss_type='NLL')
            results['QuickTest'] = (R, acc, bwt)
            print(f"\nQuick test results: ACC={acc:.2f}%, BWT={bwt:.4f}")
        else:
            print(f"Unknown experiment: {sys.argv[1]}")
            print("Usage: python forgetting_mlp.py [baseline|quick]")
    else:
        main()
