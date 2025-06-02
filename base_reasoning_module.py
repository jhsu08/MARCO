import os
import torch
import time
import traceback
from abc import ABC, abstractmethod
import numpy as np

class BaseReasoningModule(ABC):
    """
    Abstract base class for reasoning modules.
    Provides common functionality for saving and loading model states.
    """
    
    def __init__(self):
        """Initialize base reasoning module"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Performance tracking
        self.inference_time = []
        self.train_accuracy = []
    
    @abstractmethod
    def solve(self, task):
        """
        Solve a task using the reasoning approach with blackboard integration.
        
        Args:
            task: Task object to solve
            
        Returns:
            List of predictions for test examples
        """
        # This is an abstract method that should be implemented by subclasses
        # Here we provide a blueprint with blackboard logging
        
        # Log the start of reasoning
        task.blackboard.log_reasoning_step(
            module_name=self.__class__.__name__,
            details={"action": "solve_start"}
        )
        
        # Subclasses should implement their own solve logic
        # and should call the task.blackboard methods as needed
        
        raise NotImplementedError("Subclasses must implement solve()")

    # def train(self, task, optimizer, loss_fn, epochs=5, verbose=True, batch_size=1, num_workers=4, **kwargs):
    #     """
    #     Enhanced training loop implementation with task-aware batching.
        
    #     Args:
    #         task: Task object or list of tasks to train on
    #         optimizer: PyTorch optimizer
    #         loss_fn: Loss function
    #         epochs: Number of training epochs
    #         verbose: Whether to print progress messages
    #         batch_size: Number of examples to process together
    #         num_workers: Number of workers for data loading
    #         **kwargs: Additional module-specific training parameters
            
    #     Returns:
    #         Dictionary of training metrics
    #     """
    #     # Set model to training mode
    #     if hasattr(self, 'model'):
    #         self.model.train()
        
    #     # Set blackboard state for batched training
    #     is_batched = batch_size > 1
    #     if hasattr(task, 'blackboard'):
    #         original_blackboard_state = getattr(task.blackboard, 'is_batched_training', False)
    #         task.blackboard.is_batched_training = is_batched
        
    #     try:
    #         # Prepare data (can be overridden by subclasses)
    #         train_data = self._prepare_training_data(task, batch_size=batch_size, num_workers=num_workers)
            
    #         # Initialize metrics tracking
    #         metrics = {"loss": [], "accuracy": []}
            
    #         # Add grid accuracy if module supports it
    #         supports_grid_metrics = hasattr(self, "supports_grid_metrics") and self.supports_grid_metrics
    #         if supports_grid_metrics:
    #             metrics["grid_accuracy"] = []
            
    #         for epoch in range(epochs):
    #             epoch_loss = 0.0
    #             epoch_correct = 0
    #             epoch_total = 0
    #             epoch_grid_correct = 0
    #             epoch_grid_total = 0
                
    #             # Process training data by task groups
    #             task_losses = {}
    #             task_metrics = {}
                
    #             # Process training data
    #             for batch in train_data:
    #                 # Move to device if needed
    #                 if hasattr(batch, 'to') and hasattr(self, 'device'):
    #                     batch = batch.to(self.device)
                    
    #                 # Forward pass (implemented by subclasses)
    #                 # Different modules can have different return values
    #                 training_results = self._training_step(batch, optimizer, loss_fn, **kwargs)
                    
    #                 # Handle different return formats
    #                 if isinstance(training_results, tuple) and len(training_results) >= 3:
    #                     # Unpack results based on length
    #                     if len(training_results) >= 5 and supports_grid_metrics:
    #                         loss, correct, total, grid_correct, grid_total = training_results
    #                         epoch_grid_correct += grid_correct
    #                         epoch_grid_total += grid_total
    #                     else:
    #                         loss, correct, total = training_results[:3]
                        
    #                     # Group metrics by task if task_idx available in batch
    #                     if hasattr(batch, 'task_idx') and batch.task_idx is not None:
    #                         unique_tasks = torch.unique(batch.task_idx).cpu().numpy()
    #                         for task_id in unique_tasks:
    #                             if task_id not in task_losses:
    #                                 task_losses[task_id] = []
    #                                 task_metrics[task_id] = {"correct": 0, "total": 0}
    #                             task_losses[task_id].append(loss)
    #                             # Note: This is an approximation since we don't separate metrics by task
    #                             # For exact per-task metrics, modules would need to return them
    #                             task_metrics[task_id]["correct"] += correct // len(unique_tasks)
    #                             task_metrics[task_id]["total"] += total // len(unique_tasks)
                        
    #                     # Update epoch metrics
    #                     epoch_loss += loss
    #                     epoch_correct += correct
    #                     epoch_total += total
    #                 else:
    #                     # Simple case - just a loss value
    #                     epoch_loss += training_results
                
    #             # Compute epoch averages
    #             avg_loss = epoch_loss / len(train_data) if train_data and len(train_data) > 0 else 0
    #             avg_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
                
    #             # Store metrics
    #             metrics["loss"].append(avg_loss)
    #             metrics["accuracy"].append(avg_accuracy)
    #             self.train_accuracy.append(avg_accuracy)
                
    #             # Store grid metrics if supported
    #             if supports_grid_metrics and epoch_grid_total > 0:
    #                 avg_grid_accuracy = epoch_grid_correct / epoch_grid_total
    #                 metrics["grid_accuracy"].append(avg_grid_accuracy)
                
    #             # Task-specific metrics if available
    #             if task_losses:
    #                 task_avg_losses = {tid: sum(losses)/len(losses) for tid, losses in task_losses.items()}
    #                 task_avg_accuracies = {tid: m["correct"]/m["total"] if m["total"] > 0 else 0 
    #                                     for tid, m in task_metrics.items()}
                    
    #                 # Add to metrics dictionary
    #                 if "task_losses" not in metrics:
    #                     metrics["task_losses"] = []
    #                 if "task_accuracies" not in metrics:
    #                     metrics["task_accuracies"] = []
                    
    #                 metrics["task_losses"].append(task_avg_losses)
    #                 metrics["task_accuracies"].append(task_avg_accuracies)
                
    #             # Print progress if verbose
    #             if verbose:
    #                 progress_msg = f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {avg_accuracy:.4f}"
    #                 if supports_grid_metrics and "grid_accuracy" in metrics:
    #                     progress_msg += f", Grid Accuracy = {metrics['grid_accuracy'][-1]:.4f}"
    #                 print(progress_msg)
                
    #             # Additional processing (implemented by subclasses)
    #             self._after_epoch(epoch, metrics, task)
            
    #         return metrics
        
    #     finally:
    #         # Restore original blackboard state
    #         if hasattr(task, 'blackboard') and hasattr(task.blackboard, 'is_batched_training'):
    #             task.blackboard.is_batched_training = original_blackboard_state

    def prepare_gpu_dataset(self, tasks, batch_size=8, num_workers=0, shuffle=True):
        """
        Prepare a GPU-aware dataset from multiple tasks for efficient training.
        Include both train and test pairs and compute shape transformations.
        """
        # If a single task was provided, wrap it in a list
        if not isinstance(tasks, list):
            tasks = [tasks]
        
        # Import PyTorch Geometric if available
        try:
            from torch_geometric.data import Batch as PyGBatch
            from torch_scatter import scatter_mean
            pyg_available = True
        except ImportError:
            pyg_available = False
        
        # Create appropriate collate function
        if pyg_available:
            def gpu_collate(batch):
                if not batch:
                    return None
                combined_batch = PyGBatch.from_data_list(batch)
                device = self.device if hasattr(self, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                combined_batch = combined_batch.to(device)
                return combined_batch
        else:
            def gpu_collate(batch):
                if not batch:
                    return None
                return batch
        
        # Create dataset class dynamically
        class MultiTaskDataset(torch.utils.data.Dataset):
            def __init__(self, tasks, device=None):
                self.device = device
                self.data_items = []
                
                # Extract data from all tasks
                for task_idx, task_obj in enumerate(tasks):
                    # Get all train pairs for shape calculations
                    train_pairs = task_obj.train_pairs
                    test_pairs = task_obj.test_pairs
                    
                    # Use training graphs
                    for pair_idx, graph in enumerate(task_obj.train_graphs):
                        # Set task and pair indices for reference
                        graph.task_idx = torch.tensor([task_idx])
                        graph.pair_idx = torch.tensor([pair_idx])
                        
                        # Add shape transformation targets for training examples
                        if pair_idx < len(train_pairs):
                            input_grid, output_grid = train_pairs[pair_idx]
                            
                            # Calculate target shape parameters
                            input_shape = np.array(input_grid).shape
                            output_shape = np.array(output_grid).shape
                            
                            if input_shape[0] > 0 and input_shape[1] > 0:
                                # Compute transformation parameters
                                height_ratio = output_shape[0] / input_shape[0]
                                width_ratio = output_shape[1] / input_shape[1]
                                height_offset = 0  # Could use more complex calculation here if needed
                                width_offset = 0   # Could use more complex calculation here if needed
                                
                                # Add target shape to the graph
                                graph.shape_params = torch.tensor(
                                    [height_ratio, width_ratio, height_offset, width_offset],
                                    dtype=torch.float32
                                )
                                
                                # Store original and target shapes as tensor
                                graph.original_shape = torch.tensor(input_shape, dtype=torch.long)
                                graph.target_shape = torch.tensor(output_shape, dtype=torch.long)

                        
                        self.data_items.append(graph)
                    
                    # Also use test graphs
                    for pair_idx, graph in enumerate(task_obj.test_graphs):
                        graph.task_idx = torch.tensor([task_idx])
                        graph.pair_idx = torch.tensor([pair_idx + 100])  # Offset to distinguish from train

                        # Add shape transformation targets for test examples
                        if pair_idx < len(test_pairs):
                            input_grid, output_grid = test_pairs[pair_idx]
                            
                            # Calculate target shape parameters
                            input_shape = np.array(input_grid).shape
                            output_shape = np.array(output_grid).shape
                            
                            if input_shape[0] > 0 and input_shape[1] > 0:
                                # Compute transformation parameters
                                height_ratio = output_shape[0] / input_shape[0]
                                width_ratio = output_shape[1] / input_shape[1]
                                height_offset = 0  # Could use more complex calculation here if needed
                                width_offset = 0   # Could use more complex calculation here if needed
                                
                                # Add target shape to the graph
                                graph.shape_params = torch.tensor(
                                    [height_ratio, width_ratio, height_offset, width_offset],
                                    dtype=torch.float32
                                )
                        
                                # Store original and target shapes as tensor
                                graph.original_shape = torch.tensor(input_shape, dtype=torch.long)
                                graph.target_shape = torch.tensor(output_shape, dtype=torch.long)


                        self.data_items.append(graph)
                
                # Process all graphs for consistent attributes
                for graph in self.data_items:
                    # Mark as preprocessed
                    graph.preprocessed = True
                
            def __len__(self):
                return len(self.data_items)
            
            def __getitem__(self, idx):
                return self.data_items[idx]
        
        # Create dataset and dataloader
        dataset = MultiTaskDataset(tasks, device=self.device)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=gpu_collate if pyg_available else None
        )
        
        return dataloader
    
    @abstractmethod
    def _prepare_training_data(self, task, batch_size=1, num_workers=4):
        """
        Prepare training data from task.
        
        Args:
            task: Task object
            
        Returns:
            Training data (list, DataLoader, etc.)
        """
        pass
    
    @abstractmethod
    def _training_step(self, batch, optimizer, node_loss_fn, edge_loss_fn, **kwargs):
        """
        Perform a single training step with support for module-specific parameters.
        
        Args:
            batch: Training batch
            optimizer: PyTorch optimizer
            loss_fn: Primary loss function
            **kwargs: Additional module-specific parameters
            
        Returns:
            - Simple case: scalar loss value
            - Standard case: tuple of (loss, correct_predictions, total_predictions)
            - Enhanced case: tuple of (loss, correct, total, grid_correct, grid_total)
        """
        pass
    
    def _after_epoch(self, epoch, metrics, task=None):
        """
        Perform additional processing after each epoch.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
            task: The task being trained on (for blackboard access)
            
        Default implementation does nothing
        """
        pass
    
    def get_model_config(self):
        """
        Get model configuration for saving.
        
        Returns:
            Dictionary with model configuration
        """
        # Default implementation
        # Subclasses should override with specific parameters
        return {}
    
    def get_additional_state(self):
        """
        Get additional module state for saving.
        
        Returns:
            Dictionary with additional state
        """
        # Default implementation
        # Subclasses should override with specific state
        return {}
    
    def restore_additional_state(self, state):
        """
        Restore additional module state from loaded state.
        
        Args:
            state: Loaded state dictionary
            
        Returns:
            True if successful, False otherwise
        """
        # Default implementation
        # Subclasses should override with specific state restoration
        return True
    
    def create_model_from_config(self, config):
        """
        Create a model instance from configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            New model instance
        """
        # Default implementation
        # Subclasses must override this method
        raise NotImplementedError("Subclasses must implement create_model_from_config")
    
    def save_complete_state(self, save_path):
        """
        Save the complete module state.
        
        Args:
            save_path: Path to save the state
            
        Returns:
            Save path if successful
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Base state with common metrics
        state = {
            'model_state_dict': self.model.state_dict() if hasattr(self, 'model') else None,
            'inference_time': self.inference_time,
            'train_accuracy': self.train_accuracy,
            'timestamp': time.time(),
            'model_config': self.get_model_config()
        }
        
        # Add module-specific additional state
        additional_state = self.get_additional_state()
        state.update(additional_state)
        
        # Save
        try:
            torch.save(state, save_path)
            print(f"Model state saved to {save_path}")
            return save_path
        except Exception as e:
            print(f"Error saving state: {e}")
            traceback.print_exc()
            return None
    
    def load_complete_state(self, load_path):
        """
        Load the complete module state.
        
        Args:
            load_path: Path to load the state from
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(load_path):
            print(f"Error: State file not found at {load_path}")
            return False
        
        try:
            # Load state
            state = torch.load(load_path, map_location=self.device)
            
            # Validate
            if 'model_state_dict' not in state or state['model_state_dict'] is None:
                print(f"Error: Invalid state file - missing model_state_dict")
                return False
            
            # Update model if needed
            if 'model_config' in state:
                config = state['model_config']
                
                # Check if model needs to be recreated
                if hasattr(self, 'model'):
                    needs_recreation = False
                    
                    # Let subclasses decide if model recreation is needed
                    if hasattr(self, 'needs_model_recreation'):
                        needs_recreation = self.needs_model_recreation(config)
                    
                    if needs_recreation:
                        print(f"Recreating model with saved configuration")
                        self.model = self.create_model_from_config(config)
                        self.model = self.model.to(self.device)
                else:
                    # No model exists, create one
                    self.model = self.create_model_from_config(config)
                    self.model = self.model.to(self.device)
            
            # Load model weights
            if hasattr(self, 'model') and self.model is not None:
                self.model.load_state_dict(state['model_state_dict'])
            
            # Load common metrics
            if 'inference_time' in state:
                self.inference_time = state['inference_time']
            if 'train_accuracy' in state:
                self.train_accuracy = state['train_accuracy']
            
            # Restore additional state
            success = self.restore_additional_state(state)
            if not success:
                print("Warning: Failed to fully restore additional state")
            
            # Set to eval mode
            if hasattr(self, 'model') and self.model is not None:
                self.model.eval()
            
            print(f"Model state loaded from {load_path}")
            return True
            
        except Exception as e:
            print(f"Error loading state: {e}")
            traceback.print_exc()
            return False
    
    def extract_blackboard_insights(self, task):
        """
        Extract insights from the blackboard contributed by other modules.
        
        Args:
            task: Task object with blackboard
            
        Returns:
            Dictionary of insights from other modules
        """
        # Initialize empty insights container
        insights = {
            'transformations': {},
            'predicates': {},
            'patterns': {},
            'confidence_scores': {}
        }
        
        if not hasattr(task, 'blackboard'):
            return insights
            
        # The current module's name
        current_module = self.__class__.__name__
        
        # Extract from knowledge base if available
        if hasattr(task.blackboard, 'knowledge_base'):
            for key, value in task.blackboard.knowledge_base.items():
                source = task.blackboard.knowledge_sources.get(key, "unknown")
                
                # Skip own contributions
                if source == current_module:
                    continue
                    
                # Add to appropriate category
                if "transform" in key.lower():
                    insights['transformations'][key] = value
                elif "predicate" in key.lower():
                    insights['predicates'][key] = value
                elif "pattern" in key.lower():
                    insights['patterns'][key] = value
        
        # Extract confidence scores
        if hasattr(task.blackboard, 'confidence_scores'):
            insights['confidence_scores'] = {
                module: scores for module, scores in task.blackboard.confidence_scores.items()
                if module != current_module
            }
        
        return insights