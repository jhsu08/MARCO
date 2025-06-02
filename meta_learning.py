import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import traceback

class MetaLearningTrainer:
    """
    Base class for meta-learning trainers.
    Handles common functionality for episodic training.
    """
    def __init__(self, reasoning_module, device=torch.device("cuda"), log_dir="logs/meta_learning"):
        """
        Initialize the meta-learning trainer with a reasoning module
        
        Args:
            reasoning_module: A BaseReasoningModule instance (NLM, Unified, etc.)
            device: Computing device
            log_dir: Directory to save logs
        """
        self.reasoning_module = reasoning_module
        self.model = reasoning_module.model  # For backward compatibility
        self.device = device
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Track metrics
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Store blackboard insights from episodes
        self.episode_insights = []
        
    def sample_episode(self, tasks, n_support=None, n_query=None, use_blackboard=True):
        """
        Sample a meta-learning episode focusing on single-task adaptation.
        
        Args:
            tasks: List of Task objects
            n_support: Not used (we use all train examples)
            n_query: Not used (we use all test examples)
            use_blackboard: Whether to use blackboard for knowledge sharing
            
        Returns:
            Tuple of (support_set, query_set, episode_blackboard)
        """
        # Randomly select a task
        task = random.choice(tasks)
        
        # Use all train examples as support set
        support_graphs = task.train_graphs
        
        # Use all test examples as query set
        query_graphs = task.test_graphs
        
        # Initialize episode blackboard for knowledge sharing
        episode_blackboard = None
        if use_blackboard and hasattr(task, 'create_blackboard'):
            episode_blackboard = task.create_blackboard()
            
            # Copy task-specific knowledge to episode blackboard
            if hasattr(task.blackboard, 'get_all_knowledge'):
                task_knowledge = task.blackboard.get_all_knowledge()
                for source, knowledge in task_knowledge.items():
                    episode_blackboard.update_knowledge(knowledge, source=source)
        
        # Ensure all graphs have the same edge attributes
        expected_edge_types = [
            'edge_index',
            'value_edge_index',
            'region_edge_index',
            'contextual_edge_index',
            'alignment_edge_index'
        ]
        
        # Additional attributes that might be set during forward pass
        additional_attributes = [
            'edge_transformation_pred',
            'edge_type_pred',
            'edge_transformation_labels',
            'edge_type_labels',
            'node_features',
            'pattern_scores',
            'transformation_logits',
            'pattern_logits',
            'original_x'
        ]
        
        # Process all graphs to ensure attribute consistency
        all_graphs = support_graphs + query_graphs
        
        # Find which attributes are present in any graph
        present_attributes = set()
        for graph in all_graphs:
            for attr in additional_attributes:
                if hasattr(graph, attr):
                    present_attributes.add(attr)
        
        # Ensure all graphs have the same attributes
        for graph in all_graphs:
            # Add task identifier
            graph.task_id = task.task_id
            
            # Ensure edge types exist in all graphs
            for edge_type in expected_edge_types:
                if not hasattr(graph, edge_type):
                    setattr(graph, edge_type, torch.tensor([[], []], dtype=torch.long))
            
            # Ensure all additional attributes exist if they exist in any graph
            for attr in present_attributes:
                if not hasattr(graph, attr):
                    # Initialize appropriate placeholders based on attribute type
                    if attr.startswith('edge_'):
                        # Edge-related attribute
                        num_edges = graph.edge_index.size(1)
                        if attr.endswith('_pred'):
                            # For prediction tensors (create appropriate size)
                            dim = 3 if attr == 'edge_transformation_pred' else 5
                            setattr(graph, attr, torch.zeros(num_edges, dim, device=graph.edge_index.device))
                        else:
                            # For label tensors
                            setattr(graph, attr, torch.zeros(num_edges, dtype=torch.long, device=graph.edge_index.device))
                    elif attr.endswith('_logits'):
                        # For logits (assume 8 classes)
                        setattr(graph, attr, torch.zeros(graph.x.size(0), 8, device=graph.x.device))
                    elif attr == 'node_features':
                        setattr(graph, attr, torch.zeros_like(graph.x, dtype=torch.float))
                    elif attr == 'pattern_scores':
                        setattr(graph, attr, torch.zeros(graph.x.size(0), 1, device=graph.x.device))
                    elif attr == 'original_x':
                        setattr(graph, attr, graph.x.clone())
        
        # # Create batches
        # try:
        #     if support_graphs:
        #         support_batch = Batch.from_data_list(support_graphs).to(self.device)
        #     else:
        #         support_batch = None
                
        #     if query_graphs:
        #         query_batch = Batch.from_data_list(query_graphs).to(self.device)
        #     else:
        #         query_batch = None
        # except Exception as e:
        #     print(f"Error creating batches: {e}")
        #     # Debug which attributes are causing the issue
        #     if support_graphs:
        #         attr_sets = [set(dir(g)) for g in support_graphs]
        #         shared_attrs = set.intersection(*attr_sets)
        #         diff_attrs = [set(dir(g)) - shared_attrs for g in support_graphs]
        #         print(f"Differing attributes: {diff_attrs}")
        #     support_batch, query_batch = None, None
            
        # return support_batch, query_batch, episode_blackboard
        return support_graphs, query_graphs, episode_blackboard
        
    def compute_loss(self, batch, module=None):
        """
        Compute loss with shape-aware components
        """
        if batch is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if module is None:
            module = self.reasoning_module
        
        # Forward pass using the model
        if hasattr(module, 'model'):
            output = module.model(batch)
        else:
            output = module(batch)
        
        # Get predictions and targets
        predictions = output.x
        targets = batch.y
        
        # Ensure correct shapes
        if predictions.dim() != 2:
            predictions = predictions.view(-1, module.output_dim if hasattr(module, 'output_dim') else module.model.output_dim)
        
        if targets.dim() > 1:
            targets = targets.view(-1)
        
        # Get padding value
        padding_value = getattr(module, 'PADDING_VALUE', 10)
        valid_mask = targets != padding_value
        
        # Compute node value loss
        node_loss = F.cross_entropy(predictions[valid_mask], targets[valid_mask])
        total_loss = node_loss
        
        # Add shape prediction loss if available
        if hasattr(output, 'shape_params') and hasattr(batch, 'shape_params'):
            try:
                shape_params = output.shape_params
                target_shapes = batch.shape_params
                
                # Ensure compatible dimensions
                if len(shape_params.shape) == 2 and len(target_shapes.shape) == 1:
                    batch_size = shape_params.shape[0]
                    if target_shapes.shape[0] == batch_size * 4:
                        target_shapes = target_shapes.view(batch_size, 4)
                
                # Compute shape loss with custom weighting
                # Height/width ratio error is more important than offset error
                ratio_weight = 2.0
                offset_weight = 1.0
                
                # Split params
                pred_height_ratio = shape_params[:, 0]
                pred_width_ratio = shape_params[:, 1]
                pred_height_offset = shape_params[:, 2]
                pred_width_offset = shape_params[:, 3]
                
                target_height_ratio = target_shapes[:, 0]
                target_width_ratio = target_shapes[:, 1]
                target_height_offset = target_shapes[:, 2]
                target_width_offset = target_shapes[:, 3]
                
                # Weighted MSE
                height_ratio_loss = F.mse_loss(pred_height_ratio, target_height_ratio) * ratio_weight
                width_ratio_loss = F.mse_loss(pred_width_ratio, target_width_ratio) * ratio_weight
                height_offset_loss = F.mse_loss(pred_height_offset, target_height_offset) * offset_weight
                width_offset_loss = F.mse_loss(pred_width_offset, target_width_offset) * offset_weight
                
                # Combined shape loss
                shape_loss = height_ratio_loss + width_ratio_loss + height_offset_loss + width_offset_loss
                
                # Add to total loss with weight
                total_loss = total_loss + 0.25 * shape_loss
            except Exception as e:
                print(f"Error computing shape loss: {e}")
        
        # Make sure loss requires gradients
        if not total_loss.requires_grad:
            total_loss = total_loss.detach().requires_grad_(True)
        
        return total_loss
    
    def compute_accuracy(self, batch, module=None, use_solve_method=True, task=None):
        """
        Compute accuracy with proper shape handling
        """
        if batch is None:
            return 0.0, 0.0, 0.0  # Add shape_accuracy as third return value
            
        if module is None:
            module = self.reasoning_module
        
        padding_value = getattr(module, 'PADDING_VALUE', 10)
        
        # Forward pass
        with torch.no_grad():
            if use_solve_method and hasattr(module, 'solve') and task is not None:
                # Use the solve method if available and requested
                predictions = module.solve(task)
                
                # Evaluate predictions
                correct = 0
                shape_correct = 0
                content_correct = 0
                total = len(task.test_pairs)
                
                for i, ((_, expected), predicted) in enumerate(zip(task.test_pairs, predictions)):
                    expected_np = np.array(expected)
                    
                    # Check shape
                    shape_is_correct = predicted.shape == expected_np.shape
                    if shape_is_correct:
                        shape_correct += 1
                    
                    # Check content (either exact or overlapping region)
                    if shape_is_correct:
                        content_is_correct = np.array_equal(predicted, expected_np)
                    else:
                        min_rows = min(predicted.shape[0], expected_np.shape[0])
                        min_cols = min(predicted.shape[1], expected_np.shape[1])
                        content_is_correct = np.array_equal(
                            predicted[:min_rows, :min_cols],
                            expected_np[:min_rows, :min_cols]
                        )
                    
                    if content_is_correct:
                        content_correct += 1
                    
                    # Both shape and content must be correct
                    if shape_is_correct and content_is_correct:
                        correct += 1
                
                shape_accuracy = shape_correct / total if total > 0 else 0.0
                content_accuracy = content_correct / total if total > 0 else 0.0
                accuracy = correct / total if total > 0 else 0.0
                
                return accuracy, shape_accuracy, content_accuracy
            else:
                # Traditional forward pass
                if hasattr(module, 'model'):
                    output = module.model(batch)
                else:
                    output = module(batch)
                
                # Get predictions and targets
                predictions = output.x
                targets = batch.y
                
                # Apply padding mask
                valid_mask = targets != padding_value
                pred_labels = predictions.argmax(dim=1)
                
                # Node-level accuracy
                correct = (pred_labels[valid_mask] == targets[valid_mask]).sum().item()
                total = valid_mask.sum().item()
                accuracy = correct / total if total > 0 else 0.0
                
                # Extract shape predictions if available
                shape_correct = 0
                grid_correct = 0
                
                if hasattr(batch, 'batch') and hasattr(output, 'shape_params') and hasattr(batch, 'shape_params'):
                    num_graphs = batch.batch.max().item() + 1 if batch.batch.numel() > 0 else 1
                    
                    # Evaluate shape predictions
                    for i in range(min(num_graphs, output.shape_params.size(0), batch.shape_params.size(0))):
                        pred_shape = output.shape_params[i]
                        target_shape = batch.shape_params[i]
                        
                        # Consider shapes similar if within threshold
                        shape_diff = torch.abs(pred_shape - target_shape)
                        threshold = torch.tensor([0.1, 0.1, 1.0, 1.0], device=shape_diff.device)
                        shape_is_correct = torch.all(shape_diff <= threshold).item()
                        
                        if shape_is_correct:
                            shape_correct += 1
                            
                            # Check if content is also correct
                            graph_mask = batch.batch == i
                            graph_preds = pred_labels[graph_mask]
                            graph_targets = targets[graph_mask]
                            graph_valid = graph_targets != padding_value
                            
                            if graph_valid.sum() > 0:
                                content_correct = (graph_preds[graph_valid] == graph_targets[graph_valid]).all().item()
                                if content_correct:
                                    grid_correct += 1
                    
                    shape_accuracy = shape_correct / num_graphs
                    grid_accuracy = grid_correct / num_graphs
                else:
                    # Default to standard grid accuracy if no shape information
                    shape_accuracy = 0.0
                    grid_accuracy = self._compute_grid_accuracy(batch, pred_labels, targets, padding_value)
                
                return accuracy, shape_accuracy, grid_accuracy
    
    def plot_metrics(self, save_path=None):
        """Plot training metrics with cell, shape, and content accuracies"""
        plt.figure(figsize=(15, 10))
        
        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(self.train_losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        
        # Plot train accuracies
        plt.subplot(2, 2, 2)
        plt.plot(self.train_accuracies, label='Full Accuracy')
        plt.plot(self.train_shape_accuracies, label='Shape Accuracy')
        plt.plot(self.train_content_accuracies, label='Content Accuracy')
        plt.title("Training Accuracy Components")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim([0, 1.0])
        plt.legend()
        plt.grid(True)
        
        # Plot validation accuracies
        plt.subplot(2, 2, 3)
        plt.plot(self.val_accuracies, label='Full Accuracy')
        plt.plot(self.val_shape_accuracies, label='Shape Accuracy')
        plt.plot(self.val_content_accuracies, label='Content Accuracy')
        plt.title("Validation Accuracy Components")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim([0, 1.0])
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy comparison between train and val
        plt.subplot(2, 2, 4)
        plt.plot(self.train_accuracies, label='Train')
        if hasattr(self, 'val_accuracies') and self.val_accuracies:
            plt.plot(self.val_accuracies, label='Val')
        plt.title("Train vs Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim([0, 1.0])
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
        plt.close()
    
    def train_epoch(self, tasks, optimizer):
        """
        Train for one epoch.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement train_epoch")
    
    def validate(self, val_tasks, use_solve_method=True):
        """
        Validate on a set of tasks with shape-aware metrics.
        
        Args:
            val_tasks: List of validation tasks
            use_solve_method: Whether to use the module's solve method
            
        Returns:
            Tuple of (full_accuracy, shape_accuracy, content_accuracy)
        """
        if hasattr(self.reasoning_module, 'model'):
            self.reasoning_module.model.eval()
        else:
            self.reasoning_module.eval()
            
        total_full_accuracy = 0.0
        total_shape_accuracy = 0.0
        total_content_accuracy = 0.0
        total_tasks = 0
        
        for task in tqdm(val_tasks, desc="Validating model"):
            try:
                # Prepare support and query sets
                support_graphs = task.train_graphs
                query_graphs = task.test_graphs
                
                if not support_graphs or not query_graphs:
                    continue
                    
                # Create batches
                support_batch = Batch.from_data_list(support_graphs).to(self.device)
                query_batch = Batch.from_data_list(query_graphs).to(self.device)
                
                # For validation, we want the full adaptation process
                full_acc, shape_acc, content_acc = self.validate_task(support_batch, query_batch, task)
                
                total_full_accuracy += full_acc
                total_shape_accuracy += shape_acc
                total_content_accuracy += content_acc
                total_tasks += 1
                
            except Exception as e:
                print(f"Error validating task {task.task_id if hasattr(task, 'task_id') else 'unknown'}: {e}")
                traceback.print_exc()
        
        if total_tasks > 0:
            return (total_full_accuracy / total_tasks,
                    total_shape_accuracy / total_tasks,
                    total_content_accuracy / total_tasks)
        else:
            return 0.0, 0.0, 0.0
    
    def validate_task(self, support_batch, query_batch, task=None):
        """
        Validate on a single task.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement validate_task")
    
    def train(self, train_tasks, val_tasks=None, epochs=50, 
              lr=0.001, weight_decay=1e-5, save_freq=5, 
              use_solve_method=True, use_blackboard=True):
        """
        Train the model using meta-learning.
        
        Args:
            train_tasks: List of training tasks
            val_tasks: List of validation tasks
            epochs: Number of epochs
            lr: Learning rate
            weight_decay: Weight decay
            save_freq: Frequency of saving checkpoints
            use_solve_method: Whether to use the module's solve method
            use_blackboard: Whether to use blackboard for knowledge sharing
            
        Returns:
            Dictionary of training metrics
        """
        optimizer = torch.optim.Adam(
            self.reasoning_module.parameters() if hasattr(self.reasoning_module, 'parameters') 
            else self.reasoning_module.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Set blackboard integration flag if module supports it
        if hasattr(self.reasoning_module, 'use_blackboard_insights'):
            original_insight_setting = self.reasoning_module.use_blackboard_insights
            self.reasoning_module.use_blackboard_insights = use_blackboard

        # Initialize metrics tracking for shape prediction
        if not hasattr(self, 'train_shape_accuracies'):
            self.train_shape_accuracies = []
        if not hasattr(self, 'val_shape_accuracies'):
            self.val_shape_accuracies = []
            
        # Initialize metrics for content accuracy (overlapping regions)
        if not hasattr(self, 'train_content_accuracies'):
            self.train_content_accuracies = []
        if not hasattr(self, 'val_content_accuracies'):
            self.val_content_accuracies = []
        
        for epoch in range(epochs):
            # Train for one epoch
            if hasattr(self.reasoning_module, 'train'):
                self.reasoning_module.train()
            elif hasattr(self.reasoning_module, 'model'):
                self.reasoning_module.model.train()
                
            epoch_loss = self.train_epoch(train_tasks, optimizer)
            self.train_losses.append(epoch_loss)
            
            # Compute training accuracy with shape components
            train_acc, train_shape_acc, train_content_acc = self.compute_training_accuracy(train_tasks, use_solve_method)
            self.train_accuracies.append(train_acc)
            self.train_shape_accuracies.append(train_shape_acc)
            self.train_content_accuracies.append(train_content_acc)
            
            # Compute validation accuracy
            if val_tasks:
                val_acc, val_shape_acc, val_content_acc = self.validate(val_tasks, use_solve_method)
                self.val_accuracies.append(val_acc)
                self.val_shape_accuracies.append(val_shape_acc)
                self.val_content_accuracies.append(val_content_acc)
                
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, " 
                    f"Train Acc: {train_acc:.4f}, Train Shape: {train_shape_acc:.4f}, Train Content: {train_content_acc:.4f}, "
                    f"Val Acc: {val_acc:.4f}, Val Shape: {val_shape_acc:.4f}, Val Content: {val_content_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, "
                    f"Train Acc: {train_acc:.4f}, Train Shape: {train_shape_acc:.4f}, Train Content: {train_content_acc:.4f}")
                
            # Save checkpoint
            if (epoch + 1) % save_freq == 0 or epoch == epochs - 1:
                save_path = os.path.join(self.log_dir, f"model_epoch_{epoch+1}.pt")
                
                # Save model state
                if hasattr(self.reasoning_module, 'model'):
                    model_state = self.reasoning_module.model.state_dict()
                    model_type = type(self.reasoning_module).__name__
                else:
                    model_state = self.reasoning_module.state_dict()
                    model_type = type(self.reasoning_module.model).__name__
                
                # Save additional module state if available
                additional_state = {}
                if hasattr(self.reasoning_module, 'get_additional_state'):
                    additional_state = self.reasoning_module.get_additional_state()
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'model_type': model_type,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'train_accuracies': self.train_accuracies,
                    # 'train_grid_accuracies': self.train_grid_accuracies,  # Add this
                    'val_accuracies': self.val_accuracies,
                    # 'val_grid_accuracies': self.val_grid_accuracies if val_tasks else [],  # Add this
                    'additional_state': additional_state,
                    'episode_insights': self.episode_insights[-5:] if self.episode_insights else []
                }, save_path)
                
                # Plot metrics
                plot_path = os.path.join(self.log_dir, f"metrics_epoch_{epoch+1}.png")
                self.plot_metrics(plot_path)
        
        # Restore original blackboard insight setting
        if hasattr(self.reasoning_module, 'use_blackboard_insights') and original_insight_setting is not None:
            self.reasoning_module.use_blackboard_insights = original_insight_setting
                
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            # 'train_grid_accuracies': self.train_grid_accuracies,  # Add this
            'val_accuracies': self.val_accuracies,
            # 'val_grid_accuracies': self.val_grid_accuracies if val_tasks else []  # Add this
        }
    
    def compute_training_accuracy(self, tasks, use_solve_method=True):
        """
        Compute average accuracy on training tasks with shape awareness
        """
        if hasattr(self.reasoning_module, 'eval'):
            self.reasoning_module.eval()
        elif hasattr(self.reasoning_module, 'model'):
            self.reasoning_module.model.eval()
        
        # Sample a few episodes for evaluation
        total_accuracy = 0.0
        total_shape_accuracy = 0.0
        total_content_accuracy = 0.0
        n_episodes = min(10, len(tasks))
        
        for _ in range(n_episodes):
            if use_solve_method and hasattr(self.reasoning_module, 'solve'):
                # Select a random task
                task = random.choice(tasks)
                
                # Use solve method
                predictions = self.reasoning_module.solve(task)
                
                # Compute accuracies
                correct = 0
                shape_correct = 0
                content_correct = 0
                total = len(task.test_pairs)
                
                for i, ((_, expected), predicted) in enumerate(zip(task.test_pairs, predictions)):
                    expected_np = np.array(expected)
                    
                    # Check shape correctness
                    shape_is_correct = predicted.shape == expected_np.shape
                    if shape_is_correct:
                        shape_correct += 1
                    
                    # Check content correctness (either exact or overlapping region)
                    if shape_is_correct:
                        content_is_correct = np.array_equal(predicted, expected_np)
                    else:
                        min_rows = min(predicted.shape[0], expected_np.shape[0])
                        min_cols = min(predicted.shape[1], expected_np.shape[1])
                        content_is_correct = np.array_equal(
                            predicted[:min_rows, :min_cols],
                            expected_np[:min_rows, :min_cols]
                        )
                    
                    if content_is_correct:
                        content_correct += 1
                    
                    # Both shape and content must be correct for full accuracy
                    if shape_is_correct and content_is_correct:
                        correct += 1
                
                accuracy = correct / total if total > 0 else 0.0
                shape_accuracy = shape_correct / total if total > 0 else 0.0
                content_accuracy = content_correct / total if total > 0 else 0.0
                
                total_accuracy += accuracy
                total_shape_accuracy += shape_accuracy
                total_content_accuracy += content_accuracy
            else:
                # Traditional batch-based approach
                support_batch, query_batch, _ = self.sample_episode(tasks)
                if query_batch is not None:
                    cell_acc, shape_acc, content_acc = self.compute_accuracy(query_batch)
                    total_accuracy += cell_acc
                    total_shape_accuracy += shape_acc
                    total_content_accuracy += content_acc
        
        return (
            total_accuracy / n_episodes if n_episodes > 0 else 0.0,
            total_shape_accuracy / n_episodes if n_episodes > 0 else 0.0,
            total_content_accuracy / n_episodes if n_episodes > 0 else 0.0
        )

    def extract_edge_transformation_patterns(self, batch):
        """
        Extract edge transformation patterns from a batch of graphs
        
        Args:
            batch: Batch of graphs
            
        Returns:
            Dictionary of transformation patterns
        """
        patterns = {}
        
        if not hasattr(self.reasoning_module, 'model') or not hasattr(self.reasoning_module.model, 'edge_type_names'):
            return patterns
            
        edge_type_names = self.reasoning_module.model.edge_type_names
        
        try:
            # Forward pass to get predictions
            with torch.no_grad():
                output = self.reasoning_module.model(batch)
            
            # Examine edge transformation predictions if available
            if hasattr(output, 'edge_transformation_pred'):
                edge_trans_pred = output.edge_transformation_pred
                edge_type_pred = output.edge_type_pred if hasattr(output, 'edge_type_pred') else None
                
                # Analyze predictions
                for i in range(edge_trans_pred.size(0)):
                    trans_type = edge_trans_pred[i].argmax().item()
                    trans_conf = edge_trans_pred[i].max().item()
                    
                    # Get predicted edge type if available
                    edge_type = None
                    if edge_type_pred is not None:
                        edge_type_idx = edge_type_pred[i].argmax().item()
                        if edge_type_idx < len(edge_type_names):
                            edge_type = edge_type_names[edge_type_idx]
                    
                    # Store pattern
                    if trans_type not in patterns:
                        patterns[trans_type] = []
                    
                    patterns[trans_type].append({
                        'confidence': float(trans_conf),
                        'edge_type': edge_type
                    })
            
            # Aggregate pattern statistics
            for trans_type, examples in patterns.items():
                avg_conf = sum(ex['confidence'] for ex in examples) / len(examples) if examples else 0
                patterns[trans_type] = {
                    'count': len(examples),
                    'avg_confidence': avg_conf,
                    'examples': examples[:5]  # Keep only a few examples
                }
                
        except Exception as e:
            print(f"Error extracting edge transformation patterns: {e}")
            
        return patterns


class MAMLTrainer(MetaLearningTrainer):
    """
    Model-Agnostic Meta-Learning (MAML) trainer.
    
    Implements the MAML algorithm for fast adaptation to new tasks.
    """
    def __init__(self, reasoning_module, device=torch.device("cuda"), inner_lr=0.01, 
                inner_steps=3, first_order=False, log_dir="logs/maml"):
        """
        Initialize MAML trainer.
        
        Args:
            reasoning_module: A BaseReasoningModule instance
            device: Device to use
            inner_lr: Learning rate for inner loop
            inner_steps: Number of gradient steps in inner loop
            first_order: Whether to use first-order approximation
            log_dir: Directory to save logs
        """
        super().__init__(reasoning_module, device, log_dir)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
    
    def adapt_params(self, batch, params=None, create_graph=False, blackboard=None):
        """
        Adapt model parameters to a task with enhanced shape prediction capabilities.
        
        Args:
            batch: Batch of support examples
            params: Initial parameters (default: model parameters)
            create_graph: Whether to create a computation graph
            blackboard: Optional blackboard for knowledge sharing
            
        Returns:
            Adapted parameters
        """
        if batch is None:
            return params
        
        # Set blackboard on module if available
        if blackboard is not None and hasattr(self.reasoning_module, 'use_blackboard_insights'):
            self.reasoning_module.blackboard = blackboard
            
        # Get model to adapt (either the module itself or its model)
        model_to_adapt = self.reasoning_module.model if hasattr(self.reasoning_module, 'model') else self.reasoning_module
            
        if params is None:
            # Get a copy of current model parameters
            params = {name: param.clone() for name, param in model_to_adapt.named_parameters()}
        
        # Store original parameters
        original_params = {name: param.clone() for name, param in model_to_adapt.named_parameters()}
        
        # Replace model parameters
        for name, param in model_to_adapt.named_parameters():
            param.data = params[name].data
        
        # Track losses for debug and monitoring
        node_losses = []
        shape_losses = []
        edge_losses = []
        
        # Perform adaptation steps with reduced learning rate and enhanced stability
        max_norm = 5.0  # Gradient clipping threshold
        for step in range(self.inner_steps):
            # Forward pass
            try:
                # Process batch
                output = model_to_adapt(batch)
                
                # Compute node value loss
                predictions = output.x
                targets = batch.y
                
                # Ensure correct shapes
                if predictions.dim() != 2:
                    output_dim = getattr(model_to_adapt, 'output_dim', 11)
                    predictions = predictions.view(-1, output_dim)
                
                if targets.dim() > 1:
                    targets = targets.view(-1)
                
                # Apply validation mask
                padding_value = getattr(model_to_adapt, 'PADDING_VALUE', 10)
                valid_mask = targets != padding_value
                
                # Compute node classification loss
                node_loss = F.cross_entropy(predictions[valid_mask], targets[valid_mask])
                total_loss = node_loss
                node_losses.append(node_loss.item())
                
                # Add shape prediction loss if available
                if hasattr(output, 'shape_params') and hasattr(batch, 'shape_params'):
                    shape_params = output.shape_params
                    target_shapes = batch.shape_params
                    
                    # Ensure compatible dimensions
                    if len(shape_params.shape) == 2 and len(target_shapes.shape) == 1:
                        batch_size = shape_params.shape[0]
                        if target_shapes.shape[0] == batch_size * 4:
                            target_shapes = target_shapes.view(batch_size, 4)
                    
                    # Compute shape loss with weighted components
                    # Height/width ratio error is more important than offset error
                    ratio_weight = 2.0
                    offset_weight = 1.0
                    
                    # Split params
                    pred_height_ratio = shape_params[:, 0]
                    pred_width_ratio = shape_params[:, 1]
                    pred_height_offset = shape_params[:, 2]
                    pred_width_offset = shape_params[:, 3]
                    
                    target_height_ratio = target_shapes[:, 0]
                    target_width_ratio = target_shapes[:, 1]
                    target_height_offset = target_shapes[:, 2]
                    target_width_offset = target_shapes[:, 3]
                    
                    # Weighted MSE
                    height_ratio_loss = F.mse_loss(pred_height_ratio, target_height_ratio) * ratio_weight
                    width_ratio_loss = F.mse_loss(pred_width_ratio, target_width_ratio) * ratio_weight
                    height_offset_loss = F.mse_loss(pred_height_offset, target_height_offset) * offset_weight
                    width_offset_loss = F.mse_loss(pred_width_offset, target_width_offset) * offset_weight
                    
                    # Combined shape loss
                    shape_loss = height_ratio_loss + width_ratio_loss + height_offset_loss + width_offset_loss
                    shape_losses.append(shape_loss.item())
                    
                    # Add to total loss with higher weight during adaptation
                    shape_weight = 0.3  # Higher than during standard training to focus on this aspect
                    total_loss = total_loss + shape_weight * shape_loss
                
                # Add edge transformation loss if available
                if hasattr(output, 'edge_transformation_pred') and hasattr(batch, 'edge_transformation_labels'):
                    edge_preds = output.edge_transformation_pred
                    edge_labels = batch.edge_transformation_labels
                    
                    if edge_preds.size(0) == edge_labels.size(0) and edge_preds.size(0) > 0:
                        edge_loss = F.cross_entropy(edge_preds, edge_labels)
                        edge_losses.append(edge_loss.item())
                        
                        # Add to total loss with appropriate weight
                        edge_weight = 0.2
                        total_loss = total_loss + edge_weight * edge_loss
                
                # Check for NaN loss
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"Warning: NaN/Inf loss detected at step {step}. Skipping update.")
                    continue
                
                # Compute gradients with stability controls
                grads = torch.autograd.grad(
                    total_loss, model_to_adapt.parameters(), 
                    create_graph=create_graph,
                    allow_unused=True,
                    retain_graph=True
                )
                
                # Apply gradient updates with stability measures
                for i, (name, param) in enumerate(model_to_adapt.named_parameters()):
                    if grads[i] is not None:
                        # Apply gradient clipping per parameter
                        grad = grads[i].clone()
                        grad_norm = torch.norm(grad)
                        if grad_norm > max_norm:
                            grad = grad * max_norm / (grad_norm + 1e-6)
                            
                        # Skip update if gradient contains NaN/Inf
                        if torch.isnan(grad).any() or torch.isinf(grad).any():
                            print(f"Warning: NaN/Inf gradients for {name} at step {step}")
                            continue
                        
                        # Apply smaller learning rate for later steps to prevent divergence
                        # This is a form of learning rate annealing during adaptation
                        step_size = self.inner_lr # * (0.95 ** step)
                        
                        # Apply higher learning rate for shape predictor parameters to focus adaptation
                        if 'shape_predictor' in name:
                            step_size = step_size * 1.5  # Boost learning rate for shape parameters

                        # Apply higher learning rate to layers that are adapting well
                        if 'output_predictor' in name:
                            step_size = step_size * 3.0  # Much higher for these layers
                        elif 'gat_layers' in name:
                            step_size = step_size * 2.0  # Higher for these too
                        
                        # Update parameter
                        params[name] = param - step_size * grad
                        param.data = params[name].data
                    else:
                        # For parameters not used in forward pass, keep them unchanged
                        params[name] = param.clone()
                        param.data = params[name].data
                            
            except RuntimeError as e:
                print(f"Runtime error during adaptation step {step}: {e}")
                traceback.print_exc()
                break
        
        # Safety check: ensure no parameters contain NaN
        for name, param_tensor in params.items():
            if torch.isnan(param_tensor).any() or torch.isinf(param_tensor).any():
                print(f"Warning: NaN detected in adapted parameter {name}. Resetting to original.")
                params[name] = original_params[name].clone()
        
        # Restore original parameters in the model
        for name, param in model_to_adapt.named_parameters():
            param.data = original_params[name].data
        
        # Track adaptation performance
        if hasattr(self, 'adaptation_metrics'):
            self.adaptation_metrics.append({
                'node_losses': node_losses,
                'shape_losses': shape_losses if shape_losses else None,
                'edge_losses': edge_losses if edge_losses else None
            })
        else:
            self.adaptation_metrics = [{
                'node_losses': node_losses,
                'shape_losses': shape_losses if shape_losses else None,
                'edge_losses': edge_losses if edge_losses else None
            }]
            
        return params
        
    def train_epoch(self, tasks, optimizer):
        """
        Train for one epoch using MAML with improved stability and progress bar.
        
        Args:
            tasks: List of training tasks
            optimizer: Optimizer
            
        Returns:
            Average loss
        """
        total_loss = 0.0
        num_successful_tasks = 0
        
        # Create progress bar
        with tqdm(total=len(tasks), desc="Training MAML") as pbar:
            for task_idx, task in enumerate(tasks):
                try:
                    # Sample support/query sets with blackboard
                    support_graphs, query_graphs, episode_blackboard = self.sample_episode(tasks)
                    
                    if not support_graphs or not query_graphs:
                        pbar.update(1)
                        continue
                    
                    # Create batches
                    try:
                        support_batch = Batch.from_data_list(support_graphs).to(self.device)
                        query_batch = Batch.from_data_list(query_graphs).to(self.device)
                    except Exception as e:
                        print(f"Error creating batches for task {task_idx}: {e}")
                        pbar.update(1)
                        continue
                    
                    # Adapt to support set (inner loop)
                    adapted_params = self.adapt_params(
                        support_batch, 
                        create_graph=False,  # First-order approximation for stability
                        blackboard=episode_blackboard
                    )
                    
                    # Check if adaptation was successful
                    if any(torch.isnan(p).any() for p in adapted_params.values()):
                        pbar.set_postfix(status="NaN in adaptation")
                        pbar.update(1)
                        continue
                        
                    # Get model to adapt
                    model_to_adapt = self.reasoning_module.model if hasattr(self.reasoning_module, 'model') else self.reasoning_module
                    
                    # Replace model parameters with adapted parameters
                    original_params = {name: param.clone() for name, param in model_to_adapt.named_parameters()}
                    for name, param in model_to_adapt.named_parameters():
                        param.data = adapted_params[name].data
                    
                    # Evaluate on query set (outer loop)
                    try:
                        # with torch.autograd.detect_anomaly():
                        meta_loss = self.compute_loss(query_batch)
                        
                        # Skip task if loss is NaN
                        if torch.isnan(meta_loss) or torch.isinf(meta_loss):
                            pbar.set_postfix(status="NaN loss")
                            # Restore original parameters
                            for name, param in model_to_adapt.named_parameters():
                                param.data = original_params[name].data
                            pbar.update(1)
                            continue
                        
                        # Perform meta-update with gradient clipping
                        optimizer.zero_grad()
                        meta_loss.backward()
                        
                        # Apply gradient clipping to prevent explosion
                        torch.nn.utils.clip_grad_norm_(model_to_adapt.parameters(), max_norm=10.0)
                        
                        # Check for NaN gradients before updating
                        has_nan_grads = False
                        for name, param in model_to_adapt.named_parameters():
                            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                pbar.set_postfix(status=f"NaN grad in {name}")
                                has_nan_grads = True
                                break
                        
                        if not has_nan_grads:
                            optimizer.step()
                            curr_loss = meta_loss.item()
                            total_loss += curr_loss
                            num_successful_tasks += 1
                            # Update progress bar with current loss
                            pbar.set_postfix(loss=f"{curr_loss:.4f}", avg_loss=f"{total_loss/max(1, num_successful_tasks):.4f}")
                            
                    except RuntimeError as e:
                        pbar.set_postfix(status=f"Error: {str(e)[:20]}...")
                    
                    # Always restore original parameters
                    for name, param in model_to_adapt.named_parameters():
                        param.data = original_params[name].data
                    
                    # Free memory
                    del support_batch, query_batch, meta_loss, adapted_params
                    torch.cuda.empty_cache()
                    
                    # Update progress bar
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"Error processing task {task_idx}: {e}")
                    traceback.print_exc()
                    pbar.update(1)
                    continue
        
        return total_loss / max(1, num_successful_tasks)
    
    def validate_task(self, support_batch, query_batch, task=None):
        """
        Validate on a single task by adapting to support examples and then solving
        
        Args:
            support_batch: Batch of support examples
            query_batch: Batch of query examples
            task: Task object (optional)
            
        Returns:
            Tuple of (accuracy, shape_accuracy, content_accuracy)
        """
        # Adapt to support set (inner loop optimization)
        adapted_params = self.adapt_params(support_batch)
        
        # Get model to adapt
        model_to_adapt = self.reasoning_module.model if hasattr(self.reasoning_module, 'model') else self.reasoning_module
        
        # Save original parameters
        original_params = {name: param.clone() for name, param in model_to_adapt.named_parameters()}
        
        # Replace model parameters with adapted parameters
        for name, param in model_to_adapt.named_parameters():
            param.data = adapted_params[name].data
        
        # Try to use the solve method with adapted parameters if task is provided
        if task is not None and hasattr(self.reasoning_module, 'solve'):
            try:
                # Create temporary task copy to avoid modifying original
                temp_task = copy.deepcopy(task)
                
                # Solve with adapted parameters
                predictions = self.reasoning_module.solve(temp_task)
                
                # Compute accuracy with shape awareness
                correct = 0
                shape_correct = 0
                content_correct = 0
                total = len(temp_task.test_pairs)
                
                for i, ((_, expected), predicted) in enumerate(zip(temp_task.test_pairs, predictions)):
                    expected_np = np.array(expected)
                    
                    # Check shape correctness
                    shape_is_correct = predicted.shape == expected_np.shape
                    if shape_is_correct:
                        shape_correct += 1
                    
                    # Check content correctness (either exact or overlapping region)
                    if shape_is_correct:
                        content_is_correct = np.array_equal(predicted, expected_np)
                    else:
                        # For mismatched shapes, check if the overlapping region is correct
                        min_rows = min(predicted.shape[0], expected_np.shape[0])
                        min_cols = min(predicted.shape[1], expected_np.shape[1])
                        content_is_correct = np.array_equal(
                            predicted[:min_rows, :min_cols], 
                            expected_np[:min_rows, :min_cols]
                        )
                    
                    if content_is_correct:
                        content_correct += 1
                        
                    # Full accuracy requires both shape and content to be correct
                    if shape_is_correct and content_is_correct:
                        correct += 1
                
                # Restore original parameters
                for name, param in model_to_adapt.named_parameters():
                    param.data = original_params[name].data
                
                accuracy = correct / total if total > 0 else 0.0
                shape_accuracy = shape_correct / total if total > 0 else 0.0
                content_accuracy = content_correct / total if total > 0 else 0.0
                
                return accuracy, shape_accuracy, content_accuracy
                    
            except Exception as e:
                print(f"Error using solve method with adapted parameters: {e}")
                traceback.print_exc()
                # Fall back to traditional method (using the adapted parameters)
        
        # Restore original parameters
        for name, param in model_to_adapt.named_parameters():
            param.data = original_params[name].data
        
        return accuracy, shape_accuracy, content_accuracy

class ProtoNetTrainer(MetaLearningTrainer):
    """
    Prototypical Networks trainer.
    
    Implements Prototypical Networks for few-shot learning.
    """
    def __init__(self, reasoning_module, device=torch.device("cuda"), embedding_size=128, log_dir="logs/prototypical"):
        """
        Initialize Prototypical Networks trainer.
        
        Args:
            reasoning_module: A BaseReasoningModule instance
            device: Device to use
            embedding_size: Size of embeddings
            log_dir: Directory to save logs
        """
        super().__init__(reasoning_module, device, log_dir)
        self.embedding_size = embedding_size
        
        # Add embedding layer for node values
        model_to_enhance = self.reasoning_module.model if hasattr(self.reasoning_module, 'model') else self.reasoning_module
        hidden_dim = getattr(model_to_enhance, 'hidden_dim', 128)
        
        model_to_enhance.node_embedding_layer = nn.Sequential(
            nn.Linear(hidden_dim, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size)
        ).to(device)
    
    def get_embeddings(self, batch):
        """
        Get embeddings for a batch of graphs.
        
        Args:
            batch: PyG Batch object
            
        Returns:
            Embeddings tensor
        """
        if batch is None:
            return None
            
        # Forward pass through GNN
        with torch.no_grad():
            model_to_use = self.reasoning_module.model if hasattr(self.reasoning_module, 'model') else self.reasoning_module
            processed = model_to_use(batch)
        
        # Extract node features
        node_features = processed.node_features
        
        # Apply embedding layer
        embeddings = model_to_use.node_embedding_layer(node_features)
        
        # Aggregate to graph level by taking average of node embeddings
        graph_embeddings = []
        
        if hasattr(batch, 'batch'):
            num_graphs = batch.batch.max().item() + 1 if batch.batch.numel() > 0 else 1
            
            for graph_idx in range(num_graphs):
                graph_mask = batch.batch == graph_idx
                graph_nodes = embeddings[graph_mask]
                graph_embedding = graph_nodes.mean(dim=0)
                graph_embeddings.append(graph_embedding)
            
            return torch.stack(graph_embeddings)
        else:
            # Single graph
            return embeddings.mean(dim=0).unsqueeze(0)
    
    def compute_prototypes(self, embeddings, batch):
        """
        Compute class prototypes from embeddings.
        
        Args:
            embeddings: Tensor of embeddings
            batch: Batch object containing label information
            
        Returns:
            Dictionary mapping class to prototype
        """
        if embeddings is None:
            return {}
            
        # For graph-level tasks, labels are grid patterns
        # We'll use task_id as the class
        if hasattr(batch, 'task_id'):
            task_ids = batch.task_id
            
            # Ensure task_ids is a tensor, not a list
            if not isinstance(task_ids, torch.Tensor):
                task_ids = torch.tensor(task_ids, device=self.device)
                
            unique_tasks = torch.unique(task_ids)
            prototypes = {}
            
            for task_id in unique_tasks:
                task_mask = task_ids == task_id
                task_embeddings = embeddings[task_mask]
                prototypes[task_id.item()] = task_embeddings.mean(dim=0)
            
            return prototypes
        
        # For node-level tasks, use regular prototypes
        padding_value = getattr(self.reasoning_module, 'PADDING_VALUE', 10)  # Default to 10 if not defined
        
        # Ensure labels is a tensor
        if not isinstance(batch.y, torch.Tensor):
            labels = torch.tensor(batch.y, device=self.device)
        else:
            labels = batch.y
            
        unique_classes = torch.unique(labels)
        prototypes = {}
        
        for class_idx in unique_classes:
            # Skip padding class
            if class_idx == padding_value:
                continue
                
            class_mask = labels == class_idx
            class_embeddings = embeddings[class_mask]
            
            if class_embeddings.size(0) > 0:
                prototypes[class_idx.item()] = class_embeddings.mean(dim=0)
        
        return prototypes
    
    def compute_distances(self, query_embeddings, prototypes):
        """
        Compute distances between query embeddings and prototypes.
        
        Args:
            query_embeddings: Tensor of query embeddings
            prototypes: Dictionary mapping class to prototype
            
        Returns:
            Tensor of distances (num_queries, num_classes)
        """
        if query_embeddings is None or not prototypes:
            return torch.tensor([])
            
        # Stack prototypes
        prototype_tensors = torch.stack(list(prototypes.values()))
        classes = list(prototypes.keys())
        
        # Compute Euclidean distances
        num_queries = query_embeddings.size(0)
        num_prototypes = prototype_tensors.size(0)
        
        # Reshape for broadcasting
        query_embeddings = query_embeddings.unsqueeze(1).expand(num_queries, num_prototypes, -1)
        prototype_tensors = prototype_tensors.unsqueeze(0).expand(num_queries, num_prototypes, -1)
        
        # Compute squared Euclidean distance
        distances = ((query_embeddings - prototype_tensors) ** 2).sum(dim=2)
        
        return distances, classes
    
    def compute_proto_loss(self, distances_tuple, labels):
        """
        Compute prototypical network loss.
        
        Args:
            distances_tuple: Tuple of (distances, classes)
            labels: Tensor of labels
            
        Returns:
            Loss tensor
        """
        if isinstance(distances_tuple, tuple):
            distances, classes = distances_tuple
        else:
            return torch.tensor(0.0, device=self.device)
            
        if distances.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Convert distances to probabilities (negative distance for similarity)
        logits = -distances
        
        # For graph-level tasks, use task_id as the class
        if hasattr(labels, 'task_id'):
            target_indices = []
            task_ids = labels.task_id
            
            for i, task_id in enumerate(task_ids):
                try:
                    class_idx = classes.index(task_id.item())
                    target_indices.append(class_idx)
                except ValueError:
                    # Class not in prototypes, skip
                    target_indices.append(0)
            
            targets = torch.tensor(target_indices, device=self.device)
        else:
            # For node-level tasks, convert labels to indices
            padding_value = getattr(self.reasoning_module, 'PADDING_VALUE', 10)  # Default to 10 if not defined
            target_indices = []
            for label in labels:
                if label.item() == padding_value:  # Skip padding
                    target_indices.append(0)  # Default index
                else:
                    try:
                        class_idx = classes.index(label.item())
                        target_indices.append(class_idx)
                    except ValueError:
                        # Class not in prototypes, skip
                        target_indices.append(0)
            
            targets = torch.tensor(target_indices, device=self.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, targets)
        
        return loss
    
    def train_epoch(self, tasks, optimizer):
        """
        Train for one epoch using Prototypical Networks.
        
        Args:
            tasks: List of training tasks
            optimizer: Optimizer
            
        Returns:
            Average loss
        """
        total_loss = 0.0
        n_episodes = len(tasks) // 2  # Number of episodes per epoch
        
        for _ in range(n_episodes):
            # Sample support/query sets with blackboard
            support_batch, query_batch, episode_blackboard = self.sample_episode(tasks)
            
            if support_batch is None or query_batch is None:
                continue
            
            # Set blackboard on module if available
            if episode_blackboard is not None and hasattr(self.reasoning_module, 'use_blackboard_insights'):
                self.reasoning_module.blackboard = episode_blackboard
            
            # Get embeddings
            support_embeddings = self.get_embeddings(support_batch)
            query_embeddings = self.get_embeddings(query_batch)
            
            # Compute prototypes
            prototypes = self.compute_prototypes(support_embeddings, support_batch)
            
            # Compute distances
            distances = self.compute_distances(query_embeddings, prototypes)
            
            # Compute loss
            loss = self.compute_proto_loss(distances, query_batch)
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Extract patterns and insights from this episode if blackboard is available
            if episode_blackboard is not None:
                episode_insights = {
                    'edge_transformations': self.extract_edge_transformation_patterns(query_batch),
                    'blackboard_knowledge': episode_blackboard.get_all_knowledge() if hasattr(episode_blackboard, 'get_all_knowledge') else {}
                }
                self.episode_insights.append(episode_insights)
            
            total_loss += loss.item()
        
        return total_loss / n_episodes if n_episodes > 0 else 0.0
    
    def validate_task(self, support_batch, query_batch, task=None):
        """
        Validate on a single task.
        
        Args:
            support_batch: Batch of support examples
            query_batch: Batch of query examples
            task: Task object (optional)
            
        Returns:
            Accuracy on query set
        """
        # Try to use the solve method if available and task is provided
        if task is not None and hasattr(self.reasoning_module, 'solve'):
            # Create temporary task with support examples as train and query as test
            temp_task = copy.copy(task)
            
            # Set up temporary task with current examples
            # Implementation details will depend on your Task class
            if hasattr(support_batch, 'batch') and hasattr(query_batch, 'batch'):
                # Create input-output pairs from batches
                train_pairs = []
                test_pairs = []
                
                # Extract train pairs from support batch
                # This requires knowledge of your data structure
                train_pairs = [(support_batch.input_grid[i], support_batch.output_grid[i]) 
                              for i in range(len(support_batch.batch.unique()))]
                
                # Extract test pairs from query batch
                test_pairs = [(query_batch.input_grid[i], query_batch.output_grid[i]) 
                             for i in range(len(query_batch.batch.unique()))]
                
                temp_task.train_pairs = train_pairs
                temp_task.test_pairs = test_pairs
            
            # Use solve method
            try:
                predictions = self.reasoning_module.solve(temp_task)
                
                # Compute accuracy
                correct = 0
                total = len(temp_task.test_pairs)
                
                for i, ((_, expected), predicted) in enumerate(zip(temp_task.test_pairs, predictions)):
                    expected_np = np.array(expected)
                    is_correct = np.array_equal(predicted, expected_np)
                    if is_correct:
                        correct += 1
                
                accuracy = correct / total if total > 0 else 0.0
                return accuracy
            except Exception as e:
                print(f"Error using solve method for validation: {e}")
                # Fall back to traditional method
        
        # Traditional Prototypical Network approach
        # Get embeddings
        support_embeddings = self.get_embeddings(support_batch)
        query_embeddings = self.get_embeddings(query_batch)
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_embeddings, support_batch)
        
        # Compute distances
        distances, classes = self.compute_distances(query_embeddings, prototypes)
        
        # Get predictions
        predictions = torch.argmin(distances, dim=1)
        
        # Convert predictions to class labels
        class_predictions = torch.tensor([classes[p.item()] for p in predictions], 
                                        device=self.device)
        
        # Get true labels
        if hasattr(query_batch, 'task_id'):
            true_labels = query_batch.task_id
        else:
            true_labels = query_batch.y
        
        # Compute accuracy
        correct = (class_predictions == true_labels).float().mean().item()
        
        return correct
        

def run_meta_learning(reasoning_module, train_tasks, val_tasks=None, method="maml",
                      epochs=50, outer_lr=0.0001, weight_decay=1e-5, inner_lr = 0.001, inner_steps=10, log_dir="output/models/meta_learning",
                      use_solve_method=True, use_blackboard=True):
    """
    Run meta-learning on the reasoning module.
    
    Args:
        reasoning_module: A BaseReasoningModule instance
        train_tasks: List of training tasks
        val_tasks: List of validation tasks (optional)
        method: Meta-learning method ("maml" or "proto")
        epochs: Number of epochs
        lr: Learning rate
        weight_decay: Weight decay
        log_dir: Directory to save logs
        use_solve_method: Whether to use the module's solve method
        use_blackboard: Whether to use blackboard for knowledge sharing
        
    Returns:
        Trainer instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create trainer based on method
    if method.lower() == "maml":
        trainer = MAMLTrainer(
            reasoning_module=reasoning_module,
            device=reasoning_module.device,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            log_dir=os.path.join(log_dir, "maml")
        )
    elif method.lower() == "proto":
        trainer = ProtoNetTrainer(
            reasoning_module=reasoning_module,
            device=reasoning_module.device,
            embedding_size=128,
            log_dir=os.path.join(log_dir, "proto")
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Run training
    metrics = trainer.train(
        train_tasks=train_tasks,
        val_tasks=val_tasks,
        epochs=epochs,
        lr=outer_lr,
        weight_decay=weight_decay,
        use_solve_method=use_solve_method,
        use_blackboard=use_blackboard
    )
    
    # Save final metrics
    metrics_path = os.path.join(trainer.log_dir, "meta_learning_metrics.pt")
    torch.save(metrics, metrics_path)
    
    return trainer