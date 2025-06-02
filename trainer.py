import os
import json
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

from task import Task, Blackboard

# Define augmentation types
augmentation_types = [
    'rotate_90',
    'rotate_180',
    'rotate_270',
    'flip_horizontal',
    'flip_vertical',
    'value_permutation'
]

def load_tasks(directory):
    """Load tasks from directory"""
    tasks = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if "train" not in data or "test" not in data:
                        print(f"Warning: Invalid task format in {file_path}")
                        continue
                    
                    task = Task(
                        task_id=os.path.basename(file_path),
                        train_pairs=[(pair["input"], pair["output"]) for pair in data["train"]],
                        test_pairs=[(pair["input"], pair["output"]) for pair in data["test"]],
                    )
                    tasks.append(task)
    return tasks

def precompute_tasks(input_dir, output_dir="precomputed_tasks", augmentation_types=None):
    """
    Precompute tasks and store both graphs and original grids for easy loading
    
    Args:
        input_dir: Directory containing original JSON task files
        output_dir: Directory to save precomputed tasks
        augmentation_types: List of augmentation methods to apply (default: None)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all original tasks
    original_tasks_data = load_task_data(input_dir)
    print(f"Loaded {len(original_tasks_data)} original tasks from {input_dir}")
    
    # Process each task
    task_count = 0
    augmented_count = 0
    
    for task_dict in original_tasks_data:
        task_id = task_dict["task_id"]
        print(f"Processing task: {task_id}")
        
        # Create the original task
        original_task = Task(
            task_id=task_id,
            train_pairs=task_dict["train_pairs"],
            test_pairs=task_dict["test_pairs"]
        )
        
        # Save the original task with both graphs and grids
        save_task_with_grids(
            task=original_task,
            train_pairs=task_dict["train_pairs"],
            test_pairs=task_dict["test_pairs"],
            output_dir=output_dir,
            filename=f"{task_id}.pt"
        )
        task_count += 1
        
        # If augmentation is requested, generate and save augmented tasks
        if augmentation_types:
            for aug_type in augmentation_types:
                aug_task_id = f"{task_id}_{aug_type}"
                print(f"  Generating augmentation: {aug_task_id}")
                
                # Create augmented train/test pairs using the provided augment_grid function
                aug_train_pairs = []
                for input_grid, output_grid in task_dict["train_pairs"]:
                    aug_input = augment_grid(input_grid, aug_type)
                    aug_output = augment_grid(output_grid, aug_type)
                    aug_train_pairs.append((aug_input, aug_output))
                    
                aug_test_pairs = []
                for input_grid, output_grid in task_dict["test_pairs"]:
                    aug_input = augment_grid(input_grid, aug_type)
                    aug_output = augment_grid(output_grid, aug_type)
                    aug_test_pairs.append((aug_input, aug_output))
                
                # Create the augmented task with the augmented pairs
                aug_task = Task(
                    task_id=aug_task_id,
                    train_pairs=aug_train_pairs,
                    test_pairs=aug_test_pairs
                )
                
                # Save the augmented task with both graphs and grids
                save_task_with_grids(
                    task=aug_task,
                    train_pairs=aug_train_pairs,
                    test_pairs=aug_test_pairs,
                    output_dir=output_dir,
                    filename=f"{aug_task_id}.pt"
                )
                augmented_count += 1
                
    print(f"Precomputing complete: {task_count} original tasks and {augmented_count} augmented tasks saved to {output_dir}")


def load_task_data(directory):
    """Load raw task input/output grids from JSON files"""
    raw_tasks = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if "train" not in data or "test" not in data:
                        print(f"Warning: Invalid task format in {file_path}")
                        continue
                    raw_tasks.append({
                        "task_id": os.path.splitext(file)[0],
                        "train_pairs": [(pair["input"], pair["output"]) for pair in data["train"]],
                        "test_pairs": [(pair["input"], pair["output"]) for pair in data["test"]],
                    })
    return raw_tasks


def save_task_with_grids(task, train_pairs, test_pairs, output_dir, filename):
    """Save task with both precomputed graphs and original grids"""
    save_path = os.path.join(output_dir, filename)
    torch.save({
        "task_id": task.task_id,
        "train_graphs": task.train_graphs,
        "test_graphs": task.test_graphs,
        "train_targets": task.train_targets,
        "test_targets": task.test_targets,
        "train_pairs": train_pairs,
        "test_pairs": test_pairs
    }, save_path)


def load_precomputed_tasks(directory):
    """Load precomputed tasks with grids from directory"""
    tasks = []
    print(f"Loading precomputed tasks from {directory}")
    
    for root, _, files in os.walk(directory):
        pt_files = [f for f in files if f.endswith(".pt")]
        for file in tqdm(pt_files, desc=f"Loading tasks from {root}"):
            file_path = os.path.join(root, file)
            try:
                task = load_precomputed_task(file_path)
                tasks.append(task)
            except Exception as e:
                print(f"Warning: Failed to load task from {file_path}: {str(e)}")
    
    print(f"Loaded {len(tasks)} precomputed tasks")
    return tasks


def load_precomputed_task(path):
    """Load a single precomputed task from a .pt file"""
    data = torch.load(path, weights_only=False)
    task = Task.__new__(Task)  # Bypass __init__
    
    # Set the required attributes from the saved data
    task.task_id = data["task_id"]
    task.train_graphs = data["train_graphs"]
    task.test_graphs = data["test_graphs"]
    task.train_targets = data["train_targets"]
    task.test_targets = data["test_targets"]
    task.train_pairs = data["train_pairs"]
    task.test_pairs = data["test_pairs"]
    
    # Set additional required attributes
    task.edge_types = ["edge_index", "value_edge_index", "region_edge_index", 
                       "contextual_edge_index", "alignment_edge_index"]
    task.blackboard = Blackboard()
    
    return task

def augment_grid(grid, augmentation_type):
    """
    Generate augmented versions of a grid
    
    Augmentation types:
    - Rotation (90, 180, 270 degrees)
    - Horizontal/Vertical Flips
    - Color/Value Permutations
    """
    grid_np = np.array(grid)
    
    if augmentation_type == 'rotate_90':
        return np.rot90(grid_np)
    
    elif augmentation_type == 'rotate_180':
        return np.rot90(grid_np, 2)
    
    elif augmentation_type == 'rotate_270':
        return np.rot90(grid_np, 3)
    
    elif augmentation_type == 'flip_horizontal':
        return np.fliplr(grid_np)
    
    elif augmentation_type == 'flip_vertical':
        return np.flipud(grid_np)
    
    elif augmentation_type == 'value_permutation':
        # Randomly permute values while maintaining relative relationships
        unique_values = np.unique(grid_np)
        np.random.shuffle(unique_values)
        
        # Create mapping of original to new values
        value_map = {orig: new for orig, new in zip(np.unique(grid_np), unique_values)}
        return np.vectorize(value_map.get)(grid_np)
    
    return grid_np

def generate_augmented_dataset(original_tasks, augmentation_types):
    """
    Generate an expanded dataset through augmentation that preserves pattern consistency.
    For each task, create N new tasks (where N = number of augmentation types),
    with each new task having all its examples transformed in the same way.
    
    Args:
        original_tasks: List of original tasks
        augmentation_types: List of augmentation methods to apply
    
    Returns:
        List of all tasks (original + augmented)
    """
    all_tasks = list(original_tasks)  # Start with original tasks
    
    for task in original_tasks:
        task_id = task.task_id
        train_pairs = task.train_pairs
        test_pairs = task.test_pairs
        
        # For each augmentation type, create a complete new task
        for aug_type in augmentation_types:
            # Apply the same transformation to all examples
            augmented_train_pairs = []
            
            for input_grid, output_grid in train_pairs:
                # Apply the same augmentation to both input and output
                aug_input = augment_grid(input_grid, aug_type)
                aug_output = augment_grid(output_grid, aug_type)
                
                augmented_train_pairs.append((aug_input, aug_output))
            
            # Also transform the test pairs to maintain consistency
            augmented_test_pairs = []
            for input_grid, output_grid in test_pairs:
                aug_input = augment_grid(input_grid, aug_type)
                aug_output = augment_grid(output_grid, aug_type)
                
                augmented_test_pairs.append((aug_input, aug_output))
            
            # Create a new task with all examples transformed consistently
            augmented_task = Task(
                task_id=f"{task_id}_{aug_type}", 
                train_pairs=augmented_train_pairs, 
                test_pairs=augmented_test_pairs
            )
            
            all_tasks.append(augmented_task)
    
    return all_tasks

def train_model(model, tasks, num_epochs=1000, learning_rate=0.001, 
                weight_decay=1e-5, save_dir="output", model_name="model",
                batch_size=4, device="cpu"):
    import os, json, traceback
    from tqdm import tqdm
    import numpy as np
    
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = torch.optim.Adam(
        model.model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    node_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=10)
    dataloader = model._prepare_training_data(tasks, batch_size=batch_size, num_workers=0)

    history = {
        "loss": [], 
        "node_loss": [],
        "edge_trans_loss": [],
        "edge_type_loss": [],
        "grid_loss": [],
        "shape_loss": [],
        "shape_accuracy": [],
        "node_accuracy": [], 
        "grid_accuracy": [],
        "edge_trans_accuracy": [],
        "edge_type_accuracy": [],
    }

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate/10
    )

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_node_loss = 0.0
        epoch_edge_trans_loss = 0.0
        epoch_edge_type_loss = 0.0
        epoch_grid_loss = 0.0
        epoch_shape_loss = 0.0
        epoch_shape_correct = 0
        epoch_shape_total = 0
        epoch_correct = 0
        epoch_total = 0
        epoch_grid_correct = 0
        epoch_grid_total = 0
        edge_trans_correct = 0
        edge_trans_total = 0
        edge_type_correct = 0
        edge_type_total = 0
        num_batches = len(dataloader)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            try:
                batch = batch.to(device)
                training_results = model._training_step(batch, optimizer, node_loss_fn)

                (batch_loss, batch_node_loss, batch_edge_trans_loss, batch_grid_loss, batch_shape_loss,
                 batch_shape_correct, batch_shape_total,
                 batch_correct, batch_total, batch_grid_correct, batch_grid_total,
                 batch_edge_trans_correct, batch_edge_trans_total,
                 batch_edge_type_correct, batch_edge_type_total, batch_edge_type_loss) = training_results

                epoch_loss += batch_loss / num_batches
                epoch_node_loss += batch_node_loss / num_batches
                epoch_edge_trans_loss += batch_edge_trans_loss / num_batches
                epoch_edge_type_loss += batch_edge_type_loss / num_batches
                epoch_grid_loss += batch_grid_loss / num_batches
                epoch_shape_loss += batch_shape_loss / num_batches
                epoch_shape_correct += batch_shape_correct
                epoch_shape_total += batch_shape_total
                epoch_correct += batch_correct
                epoch_total += batch_total
                epoch_grid_correct += batch_grid_correct
                epoch_grid_total += batch_grid_total
                edge_trans_correct += batch_edge_trans_correct
                edge_trans_total += batch_edge_trans_total
                edge_type_correct += batch_edge_type_correct
                edge_type_total += batch_edge_type_total

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                traceback.print_exc()

        # Accuracy computations
        node_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        grid_acc = epoch_grid_correct / epoch_grid_total if epoch_grid_total > 0 else 0.0
        shape_acc = epoch_shape_correct / epoch_shape_total if epoch_shape_total > 0 else 0.0
        edge_trans_acc = edge_trans_correct / edge_trans_total if edge_trans_total > 0 else 0.0
        edge_type_acc = edge_type_correct / edge_type_total if edge_type_total > 0 else 0.0

        # Update history
        history["loss"].append(epoch_loss)
        history["node_loss"].append(epoch_node_loss)
        history["edge_trans_loss"].append(epoch_edge_trans_loss)
        history["edge_type_loss"].append(epoch_edge_type_loss)
        history["grid_loss"].append(epoch_grid_loss)
        history["shape_loss"].append(epoch_shape_loss)
        history["node_accuracy"].append(node_acc)
        history["grid_accuracy"].append(grid_acc)
        history["shape_accuracy"].append(shape_acc)
        history["edge_trans_accuracy"].append(edge_trans_acc)
        history["edge_type_accuracy"].append(edge_type_acc)

        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Loss: {epoch_loss:.4f} (Node: {epoch_node_loss:.4f}, EdgeTrans: {epoch_edge_trans_loss:.4f}, EdgeType: {epoch_edge_type_loss:.4f}, Grid: {epoch_grid_loss:.4f}, Shape: {epoch_shape_loss:.4f}), "
              f"Node Acc: {node_acc:.4f}, Grid Acc: {grid_acc:.4f}, Shape Acc: {shape_acc:4f}, EdgeTrans Acc: {edge_trans_acc:.4f}, EdgeType Acc: {edge_type_acc:.4f}")

        # Save checkpoint and plot
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            path = os.path.join(save_dir, f"{model_name}_epoch_{epoch+1}.pt")
            model.save_complete_state(path)
            print(f"Checkpoint saved to {path}")

            plot_training_metrics(history, title=f"{model_name} Training (Epoch {epoch+1})",
                                  save_path=os.path.join(save_dir, f"{model_name}_training_epoch_{epoch+1}.png"))

        model._after_epoch(epoch, history, tasks[0] if tasks else None)

    # Save final model and history
    final_model_path = os.path.join(save_dir, f"{model_name}_final.pt")
    model.save_complete_state(final_model_path)
    print(f"Final model saved to {final_model_path}")

    history_path = os.path.join(save_dir, f"{model_name}_history.json")
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f, indent=2)
    print(f"History saved to {history_path}")

    return model, history


def plot_training_metrics(metrics, title="Training Metrics", save_path=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(21, 6))

    # Total Loss
    plt.subplot(1, 3, 1)
    plt.plot(metrics["loss"])
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # Component Losses
    plt.subplot(1, 3, 2)
    for loss_key in ["node_loss", "grid_loss", "shape_loss", "edge_trans_loss", "edge_type_loss"]:
        if loss_key in metrics:
            plt.plot(metrics[loss_key], label=loss_key.replace('_', ' ').title())
    plt.title("Loss Components")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy Metrics
    plt.subplot(1, 3, 3)
    for acc_key in ["node_accuracy", "grid_accuracy", "shape_accuracy", "edge_trans_accuracy", "edge_type_accuracy"]:
        if acc_key in metrics:
            plt.plot(metrics[acc_key], label=acc_key.replace('_', ' ').title())
    plt.title("Accuracy Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def preprocess_task_graphs(tasks, padding_value=10):
    """
    Preprocess all graphs in the given tasks once, instead of during each solve call.
    
    Args:
        tasks: List of Task objects
        padding_value: Padding value for standardizing dimensions
        
    Returns:
        The tasks with preprocessed graphs
    """
    import torch
    
    expected_dim = 3  # Standardized input dimension
    
    for task in tasks:
        # Preprocess test graphs
        for test_graph in task.test_graphs:
            if hasattr(test_graph, 'x'):
                # Convert from one-hot to class labels if needed
                if test_graph.x.dim() == 2 and test_graph.x.size(1) == 11:
                    test_graph.x = test_graph.x.argmax(dim=1)

                # Ensure x is long and 2D
                test_graph.x = test_graph.x.long()
                if test_graph.x.dim() == 1:
                    test_graph.x = test_graph.x.unsqueeze(1)  # Shape: (nodes, 1)

                # Standardize shape to (nodes, expected_dim)
                if test_graph.x.size(1) < expected_dim:
                    pad = torch.full((test_graph.x.size(0), expected_dim), padding_value, dtype=torch.long)
                    pad[:, :test_graph.x.size(1)] = test_graph.x
                    test_graph.x = pad
                elif test_graph.x.size(1) > expected_dim:
                    test_graph.x = test_graph.x[:, :expected_dim]

                # Extract positional info
                test_graph.pos = test_graph.x[:, 1:3].float() if expected_dim >= 3 else None
                test_graph.x = test_graph.x[:, 0].long().unsqueeze(1)  # Final x: shape (nodes, 1)
                
                # Mark as preprocessed
                test_graph.preprocessed = True
        
        # If the task also has train graphs, preprocess them too
        if hasattr(task, 'train_graphs'):
            for train_graph in task.train_graphs:
                if hasattr(train_graph, 'x'):
                    # Convert from one-hot to class labels if needed
                    if train_graph.x.dim() == 2 and train_graph.x.size(1) == 11:
                        train_graph.x = train_graph.x.argmax(dim=1)
    
                    # Ensure x is long and 2D
                    train_graph.x = train_graph.x.long()
                    if train_graph.x.dim() == 1:
                        train_graph.x = train_graph.x.unsqueeze(1)  # Shape: (nodes, 1)
    
                    # Standardize shape to (nodes, expected_dim)
                    if train_graph.x.size(1) < expected_dim:
                        pad = torch.full((train_graph.x.size(0), expected_dim), padding_value, dtype=torch.long)
                        pad[:, :train_graph.x.size(1)] = train_graph.x
                        train_graph.x = pad
                    elif train_graph.x.size(1) > expected_dim:
                        train_graph.x = train_graph.x[:, :expected_dim]
    
                    # Extract positional info
                    train_graph.pos = train_graph.x[:, 1:3].float() if expected_dim >= 3 else None
                    train_graph.x = train_graph.x[:, 0].long().unsqueeze(1)  # Final x: shape (nodes, 1)
                    
                    # Mark as preprocessed
                    train_graph.preprocessed = True
    
    return tasks