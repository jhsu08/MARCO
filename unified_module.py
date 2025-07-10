from base_reasoning_module import BaseReasoningModule

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, BatchNorm
from torch_geometric.data import Data
import time
import numpy as np
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
import os
import traceback
    
class UnifiedReasoningGNN(nn.Module):
    """
    Unified Graph Neural Network for spatial and transformation reasoning
    that leverages multiple edge types and attention mechanisms.
    """
    def __init__(self, 
                 input_dim=3, 
                 hidden_dim=128, 
                 output_dim=11, 
                 num_transformation_types=8,
                 num_pattern_types=8,
                 num_edge_transformation_types=3):  # Added parameter for edge transformation types
        super().__init__()
        
        # Configuration parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_transformation_types = num_transformation_types
        self.num_pattern_types = num_pattern_types
        self.num_edge_transformation_types = num_edge_transformation_types  # Store new parameter
        
        # Node embedding for categorical values (0-10 + padding)
        self.node_embedding = nn.Embedding(11, hidden_dim // 2)
        
        # Spatial and positional feature encoding
        self.position_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )
        
        # Define edge type names for consistency
        self.edge_type_names = [
            'edge_index',  # Spatial edges
            'value_edge_index',
            'region_edge_index', 
            'contextual_edge_index', 
            'alignment_edge_index'
        ]
        
        # Edge type embeddings to capture semantic differences
        self.edge_type_embedding = nn.Embedding(len(self.edge_type_names), hidden_dim // 4)
        
        # Multi-head Graph Attention Layers
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=4, concat=True),      # Output: hidden_dim * 4
            GATConv(hidden_dim * 4, hidden_dim, heads=2, concat=False), # Output: hidden_dim
            GATConv(hidden_dim, hidden_dim, heads=1, concat=True),      # Output: hidden_dim
        ])
        
        # Batch normalization for stability
        self.batch_norms = nn.ModuleList([
            BatchNorm(hidden_dim * 4),
            BatchNorm(hidden_dim),
            BatchNorm(hidden_dim)
        ])
        
        # Transformation detection head
        self.transformation_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_transformation_types)
        )
        
        # Spatial pattern recognition head
        self.pattern_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_pattern_types)
        )
        
        # Node value prediction head
        self.node_value_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Pattern significance detector
        self.pattern_significance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Edge type classifier (multi-class)
        self.edge_type_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, len(self.edge_type_names))
        )
        
        # Edge transformation predictor
        self.edge_transformation_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_edge_transformation_types)  # typically 3
        )

        # To allow residual connection between gat1 output and gat3 output
        self.linear_rescale = nn.Linear(hidden_dim * 4, hidden_dim)

        # Shape predictor
        self.shape_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 4)  # [height_ratio, width_ratio, height_offset, width_offset]
        )
    
    def forward(self, data):
        """
        Process graph with multiple edge types and attention.

        Args:
            data: PyG Data object with multiple edge indices

        Returns:
            Processed graph data with additional attributes
        """
        # Extract node values and spatial features
        x = data.x.clone()
        if x.dim() == 2 and x.size(1) >= 3:
            node_values = x[:, 0].long().clamp(0, 10)
            spatial_feats = x[:, 1:3]
        else:
            node_values = x.view(-1).long().clamp(0, 10)
            spatial_feats = torch.zeros(x.size(0), 2, device=x.device)

        # Embed node values and spatial features
        value_embedding = self.node_embedding(node_values)
        position_encoding = self.position_encoder(spatial_feats)

        # Combine initial node features
        combined_features = torch.cat([value_embedding, position_encoding], dim=1)

        # Use candidate_edge_index if available (for transformation prediction)
        edge_index = getattr(data, "candidate_edge_index", data.edge_index)

        # First GAT layer
        x1 = self.gat_layers[0](combined_features, edge_index)
        x1 = self.batch_norms[0](x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        
        # Second GAT layer
        x2 = self.gat_layers[1](x1, edge_index)
        x2 = self.batch_norms[1](x2)
        x2 = F.relu(x2)
        
        # Third GAT layer
        x3 = self.gat_layers[2](x2, edge_index)
        x3 = self.batch_norms[2](x3)
        x3 = F.relu(x3)
        
        # Skip connection (from x1 to x3)
        x_res = self.linear_rescale(x1)     # Project x1 to match x3's dimensions
        x = F.relu(x3 + x_res)              # Final feature map after skip connection

        # Predictions
        transformation_logits = self.transformation_detector(x)
        pattern_logits = self.pattern_classifier(x)
        node_value_predictions = self.node_value_predictor(x)
        pattern_scores = self.pattern_significance(x)

        # Create result object for all predictions
        result = data.clone()
        
        # Store node-level outputs
        result.transformation_logits = transformation_logits
        result.pattern_logits = pattern_logits
        result.node_features = x
        result.pattern_scores = pattern_scores
        result.x = node_value_predictions
        result.original_x = node_values

        # Edge transformation prediction
        src, dst = edge_index[0], edge_index[1]
        src_features = x[src]
        dst_features = x[dst]
        edge_feats = torch.cat([src_features, dst_features], dim=1)

        edge_trans_logits = self.edge_transformation_predictor(edge_feats)
        result.edge_transformation_pred = edge_trans_logits  # Store in result

        # Edge type prediction
        edge_type_labels = getattr(data, "candidate_edge_type", getattr(data, "edge_type", None))
        if edge_type_labels is not None and edge_type_labels.size(0) == edge_feats.size(0):
            result.edge_type_labels = edge_type_labels
            result.edge_type_pred = self.edge_type_classifier(edge_feats)
        else:
            print(f"[WARNING] Skipping edge type prediction: edge_type size {edge_type_labels.size(0) if edge_type_labels is not None else 'None'} != edge_feats {edge_feats.size(0)}")

        # Add shape prediction using global graph embedding
        if hasattr(data, 'batch'):
            graph_embedding = scatter_mean(x, data.batch, dim=0)
        else:
            graph_embedding = x.mean(dim=0, keepdim=True)
        
        # Predict shape transformation
        shape_params = self.shape_predictor(graph_embedding)
        shape_params = shape_params.view(-1, 4)
        result.shape_params = shape_params  # The critical change - store in result

        return result
    
class UnifiedReasoningModule(BaseReasoningModule):
    """
    Simplified unified reasoning module 
    that leverages multi-edge graph representation with edge transformation prediction
    """
    
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=11, 
                 num_edge_transformation_types=3, device="cpu", use_blackboard_insights=False):
        """Initialize the unified reasoning module"""
        super().__init__()

        # Module supports grid metrics
        self.supports_grid_metrics = True
        self.use_blackboard_insights = use_blackboard_insights
        
        # Store configuration
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_edge_transformation_types = num_edge_transformation_types  # Added parameter
        self.device = device
        self.PADDING_VALUE = 10  # Define padding value for consistency
        
        # Create the core reasoning model
        self.model = self.create_model_from_config({
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'num_edge_transformation_types': num_edge_transformation_types  # Pass to model creation
        })
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Performance tracking
        self.inference_time = []
    
    def create_model_from_config(self, config):
        """
        Create reasoning model from configuration
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            New reasoning model instance
        """
        return UnifiedReasoningGNN(
            input_dim=config.get('input_dim', 3),
            hidden_dim=config.get('hidden_dim', 128),  # Note the increased hidden_dim
            output_dim=config.get('output_dim', 11),
            num_transformation_types=config.get('num_transformation_types', 8),
            num_pattern_types=config.get('num_pattern_types', 8),
            num_edge_transformation_types=config.get('num_edge_transformation_types', 3)  # Added param
        )
    
    def get_additional_state(self):
        """Get additional module state for saving"""
        return {
            'inference_time': self.inference_time
        }
    
    def restore_additional_state(self, state):
        """Restore additional module state"""
        try:
            if 'inference_time' in state:
                self.inference_time = state['inference_time']
            return True
        except Exception as e:
            print(f"Error restoring additional state: {e}")
            return False
        
    # def _after_epoch(self, epoch, metrics, task=None):
    #     """
    #     Process patterns and transformations after each epoch
        
    #     Args:
    #         epoch: Current epoch
    #         metrics: Epoch training metrics
    #         task: Task being trained
    #     """
    #     if task is not None:
    #         # Extract patterns and transformations
    #         epoch_analysis = []
            
    #         for i, (input_grid, output_grid) in enumerate(task.train_pairs):
    #             try:
    #                 # Convert to graphs
    #                 input_graph = task.grid_to_graph(
    #                     input_grid, 
    #                     logical=True,
    #                     add_value_based_edges=True,
    #                     add_region_edges=True,
    #                     add_contextual_edges=True,
    #                     add_alignment_edges=True
    #                 )
    #                 output_graph = task.grid_to_graph(
    #                     output_grid, 
    #                     logical=True,
    #                     add_value_based_edges=True,
    #                     add_region_edges=True,
    #                     add_contextual_edges=True,
    #                     add_alignment_edges=True
    #                 )
                    
    #                 # Move graphs to device
    #                 input_graph = input_graph.to(self.device)
    #                 output_graph = output_graph.to(self.device)
                    
    #                 # Process graphs
    #                 with torch.no_grad():
    #                     processed_in = self.model(input_graph)
    #                     processed_out = self.model(output_graph)
                        
    #                     # Track transformations
    #                     transformations = {}
    #                     for node_idx in range(processed_in.x.size(0)):
    #                         input_val = processed_in.original_x[node_idx].item()
    #                         output_val = processed_out.original_x[node_idx].item()
                            
    #                         if input_val != output_val:
    #                             transformations[node_idx] = {
    #                                 "change_type": "node_color",
    #                                 "from": input_val,
    #                                 "to": output_val,
    #                                 "transformation_type": int(processed_in.transformation_logits[node_idx].argmax().item()),
    #                                 "transformation_confidence": float(processed_in.transformation_logits[node_idx].max().item())
    #                             }
                        
    #                     # Track edge transformations as well
    #                     edge_transformations = {}
    #                     if hasattr(processed_in, 'edge_transformation_pred'):
    #                         print("[DEBUG] Edge transformation prediction available")
    #                         edge_pred = processed_in.edge_transformation_pred
    #                         edge_indices = []
                            
    #                         # Collect all edge indices
    #                         for edge_type in self.model.edge_type_names:
    #                             if hasattr(input_graph, edge_type):
    #                                 edge_idx = getattr(input_graph, edge_type)
    #                                 if edge_idx.numel() > 0:
    #                                     # Convert to tuple form for dictionary keys
    #                                     for idx in range(edge_idx.size(1)):
    #                                         src, dst = edge_idx[0, idx].item(), edge_idx[1, idx].item()
    #                                         edge_indices.append((src, dst, edge_type))
                            
    #                         # Match predictions to edge indices
    #                         if len(edge_indices) == edge_pred.size(0):
    #                             for i, (src, dst, edge_type) in enumerate(edge_indices):
    #                                 trans_type = edge_pred[i].argmax().item()
    #                                 trans_conf = edge_pred[i].max().item()
                                    
    #                                 edge_transformations[(src, dst)] = {
    #                                     "edge_type": edge_type,
    #                                     "transformation_type": int(trans_type),
    #                                     "transformation_confidence": float(trans_conf)
    #                                 }
                        
    #                     # Compile analysis for this example
    #                     epoch_analysis.append({
    #                         "example": i,
    #                         "node_transformations": transformations,
    #                         "edge_transformations": edge_transformations,  # Added
    #                         "transformation_logits": processed_in.transformation_logits.cpu().numpy(),
    #                         "pattern_logits": processed_in.pattern_logits.cpu().numpy(),
    #                         "pattern_scores": processed_in.pattern_scores.cpu().numpy()
    #                     })
                            
    #             except Exception as e:
    #                 print(f"Error extracting patterns for example {i}: {e}")
            
    #         # Store patterns and transformations in module
    #         if hasattr(self, 'detected_patterns'):
    #             self.detected_patterns.append({
    #                 'epoch': epoch,
    #                 'patterns': epoch_analysis
    #             })
            
    #         if hasattr(self, 'detected_transformations'):
    #             self.detected_transformations.append({
    #                 'epoch': epoch,
    #                 'transformations': epoch_analysis
    #             })
            
    #         # Store in blackboard if available
    #         if hasattr(task, 'blackboard') and hasattr(task.blackboard, 'update_knowledge'):
    #             task.blackboard.update_knowledge(
    #                 {
    #                     'epoch': epoch, 
    #                     'training_analysis': epoch_analysis
    #                 },
    #                 source='epoch_unified_reasoning'
    #             )
    def solve(self, task):
        """
        Solve a task using unified reasoning with blackboard insights and pre-processed graphs
        with proper shape prediction handling
        
        Args:
            task: Task object with blackboard
            
        Returns:
            List of predictions for test examples
        """
        predictions = []
        start_time = time.time()
        
        try:
            # Initialize insights as empty
            insights = {}
            
            # Only retrieve insights if the feature is enabled
            if self.use_blackboard_insights and hasattr(task, 'blackboard') and hasattr(task.blackboard, 'get_insights_for_module'):
                module_name = self.__class__.__name__
                insights = task.blackboard.get_insights_for_module(module_name)
            
            # Process each test example using pre-processed test graphs
            for i, test_graph in enumerate(task.test_graphs):
                # Get the original test input for reference
                test_input, _ = task.test_pairs[i]
                
                # Move graph to device
                test_graph = test_graph.to(self.device)
                
                # Apply any transformation patterns from insights if available
                if insights and 'transformations' in insights:
                    # Extract relevant transformation patterns
                    patterns = insights.get('transformations', {})
                    if patterns:
                        # Log that we're using transformation patterns
                        print(f"Using {len(patterns)} transformation patterns from blackboard")
                        # The model will use these patterns during inference
                
                # Process input 
                with torch.no_grad():
                    # For single graphs, add an explicit batch dimension to avoid scatter_mean issues
                    if not hasattr(test_graph, 'batch') or test_graph.batch is None:
                        # Create a batch attribute with all zeros (single graph)
                        test_graph.batch = torch.zeros(test_graph.x.size(0), dtype=torch.long, device=test_graph.x.device)
                    
                    processed = self.model(test_graph)
                
                # Extract the predicted shape from the model
                predicted_shape = np.array(test_input).shape  # Default to input shape if no prediction available
                shape_info = None
                
                if hasattr(processed, 'shape_params'):
                    try:
                        # Get the predicted shape parameters
                        shape_params = processed.shape_params.cpu().numpy()[0]  # [height_ratio, width_ratio, height_offset, width_offset]
                        
                        # Calculate new dimensions
                        height_ratio, width_ratio, height_offset, width_offset = shape_params
                        predicted_height = max(1, int(round(np.array(test_input).shape[0] * height_ratio + height_offset)))
                        predicted_width = max(1, int(round(np.array(test_input).shape[1] * width_ratio + width_offset)))
                        predicted_shape = (predicted_height, predicted_width)
                        
                        shape_info = {
                            "predicted_params": shape_params.tolist(),
                            "input_shape": np.array(test_input).shape,
                            "predicted_shape": predicted_shape
                        }
                    except Exception as e:
                        print(f"Error in shape prediction: {e}")
                        predicted_shape = np.array(test_input).shape
                        shape_info = {
                            "error": str(e),
                            "input_shape": np.array(test_input).shape,
                            "predicted_shape": np.array(test_input).shape
                        }
                
                # Convert to grid with the predicted shape
                prediction = task.graph_to_grid(
                    processed, 
                    predicted_shape
                )
                
                # Enhance prediction with blackboard insights if available
                if insights and insights.get('high_confidence_predictions'):
                    # Find relevant predictions for this test example
                    for pred_info in insights['high_confidence_predictions']:
                        other_predictions = pred_info['prediction']
                        
                        # Check if this prediction applies to our current example
                        if isinstance(other_predictions, list) and i < len(other_predictions):
                            other_pred = other_predictions[i]
                            
                            # If it's a high-confidence prediction from another module
                            if pred_info['confidence'] > 0.85:
                                if other_pred.shape == prediction.shape:
                                    # Create a weighted blend based on confidence
                                    prediction = self.blend_predictions(prediction, other_pred, 
                                                weight_self=0.7, weight_other=0.3)
                                    
                                    # Log that we enhanced the prediction
                                    print(f"Enhanced prediction using insights from {pred_info['module']}")
                
                predictions.append(prediction)
                
                # Log reasoning step and include edge transformation and shape prediction info
                confidence = float(processed.transformation_logits.max(dim=1)[0].mean().item())
                
                # Extract edge transformation predictions if available
                edge_transformation_info = None
                if hasattr(processed, 'edge_transformation_pred'):
                    edge_transformation_info = {
                        "mean_confidence": float(processed.edge_transformation_pred.max(dim=1)[0].mean().item()),
                        "num_edges": processed.edge_transformation_pred.size(0)
                    }
                
                task.log_reasoning_step(
                    module_name="UnifiedReasoningModule",
                    prediction=prediction,
                    confidence=confidence,
                    time_taken=time.time() - start_time,
                    details={
                        "edge_transformations": edge_transformation_info,
                        "shape_prediction": shape_info,  # Add shape prediction info
                        "used_blackboard_insights": bool(insights and (insights.get('transformations') or 
                                                    insights.get('high_confidence_predictions'))),
                        "used_precomputed_graphs": True
                    }
                )
            
            # Track inference time
            inference_time = time.time() - start_time
            self.inference_time.append(inference_time)
            
            return predictions
            
        except Exception as e:
            print(f"Error in UnifiedReasoningModule.solve: {e}")
            traceback.print_exc()
            
            # Fallback to input copying
            for test_input, _ in task.test_pairs:
                predictions.append(np.array(test_input))
            
            return predictions
    # def solve(self, task):
    #     """
    #     Solve a task using unified reasoning with blackboard insights and pre-processed graphs
        
    #     Args:
    #         task: Task object with blackboard
            
    #     Returns:
    #         List of predictions for test examples
    #     """
    #     predictions = []
    #     start_time = time.time()
        
    #     try:
    #         # Initialize insights as empty
    #         insights = {}
            
    #         # Only retrieve insights if the feature is enabled
    #         if self.use_blackboard_insights and hasattr(task, 'blackboard') and hasattr(task.blackboard, 'get_insights_for_module'):
    #             module_name = self.__class__.__name__
    #             insights = task.blackboard.get_insights_for_module(module_name)
            
    #         # Process each test example using pre-processed test graphs
    #         for i, test_graph in enumerate(task.test_graphs):
    #             # Get the original test input for reference
    #             test_input, _ = task.test_pairs[i]
                
    #             # Move graph to device
    #             test_graph = test_graph.to(self.device)
                
    #             # Apply any transformation patterns from insights if available
    #             if insights and 'transformations' in insights:
    #                 # Extract relevant transformation patterns
    #                 patterns = insights.get('transformations', {})
    #                 if patterns:
    #                     # Log that we're using transformation patterns
    #                     print(f"Using {len(patterns)} transformation patterns from blackboard")
    #                     # The model will use these patterns during inference
                
    #             # Process input 
    #             with torch.no_grad():
    #                 processed = self.model(test_graph)
                
    #             # Convert to grid
    #             prediction = task.graph_to_grid(
    #                 processed, 
    #                 np.array(test_input).shape
    #             )
                
    #             # Enhance prediction with blackboard insights if available
    #             if insights and insights.get('high_confidence_predictions'):
    #                 # Find relevant predictions for this test example
    #                 for pred_info in insights['high_confidence_predictions']:
    #                     other_predictions = pred_info['prediction']
                        
    #                     # Check if this prediction applies to our current example
    #                     if isinstance(other_predictions, list) and i < len(other_predictions):
    #                         other_pred = other_predictions[i]
                            
    #                         # If it's a high-confidence prediction from another module
    #                         if pred_info['confidence'] > 0.85:
    #                             if other_pred.shape == prediction.shape:
    #                                 # Create a weighted blend based on confidence
    #                                 prediction = self.blend_predictions(prediction, other_pred, 
    #                                             weight_self=0.7, weight_other=0.3)
                                    
    #                                 # Log that we enhanced the prediction
    #                                 print(f"Enhanced prediction using insights from {pred_info['module']}")
                
    #             predictions.append(prediction)
                
    #             # Log reasoning step and include edge transformation info
    #             confidence = float(processed.transformation_logits.max(dim=1)[0].mean().item())
                
    #             # Extract edge transformation predictions if available
    #             edge_transformation_info = None
    #             if hasattr(processed, 'edge_transformation_pred'):
    #                 edge_transformation_info = {
    #                     "mean_confidence": float(processed.edge_transformation_pred.max(dim=1)[0].mean().item()),
    #                     "num_edges": processed.edge_transformation_pred.size(0)
    #                 }
                
    #             task.log_reasoning_step(
    #                 module_name="UnifiedReasoningModule",
    #                 prediction=prediction,
    #                 confidence=confidence,
    #                 time_taken=time.time() - start_time,
    #                 details={
    #                     "edge_transformations": edge_transformation_info,
    #                     "used_blackboard_insights": bool(insights and insights.get('transformations')),
    #                     "used_precomputed_graphs": True
    #                 }
    #             )
            
    #         # Track inference time
    #         inference_time = time.time() - start_time
    #         self.inference_time.append(inference_time)
            
    #         return predictions
            
    #     except Exception as e:
    #         print(f"Error in UnifiedReasoningModule.solve: {e}")
    #         traceback.print_exc()
            
    #         # Fallback to input copying
    #         for test_input, _ in task.test_pairs:
    #             predictions.append(np.array(test_input))
            
    #         return predictions

    # def blend_predictions(self, pred1, pred2, weight_self=0.5, weight_other=0.5):
    #     """
    #     Blend two grid predictions with weights
        
    #     Args:
    #         pred1: First prediction (self)
    #         pred2: Second prediction (other)
    #         weight_self: Weight for first prediction
    #         weight_other: Weight for second prediction
            
    #     Returns:
    #         Blended prediction
    #     """
    #     # For categorical data like grids, we need special handling
    #     # Here we'll implement a simple cell-by-cell weighted selection
        
    #     # Ensure same shape
    #     if pred1.shape != pred2.shape:
    #         return pred1  # Can't blend different shapes
        
    #     # Initialize with a copy of pred1
    #     blended = pred1.copy()
        
    #     # Find cells where predictions differ
    #     different_cells = (pred1 != pred2)
        
    #     # If no differences, return original
    #     if not np.any(different_cells):
    #         return blended
        
    #     # For cells that differ, use the prediction from the module with higher weight
    #     if weight_other > weight_self:
    #         blended[different_cells] = pred2[different_cells]
        
    #     return blended
    def blend_predictions(self, pred1, pred2, weight_self=0.5, weight_other=0.5):
        """
        Blend two grid predictions with weights, handling different shapes intelligently
        
        Args:
            pred1: First prediction (self)
            pred2: Second prediction (other)
            weight_self: Weight for first prediction
            weight_other: Weight for second prediction
            
        Returns:
            Blended prediction
        """
        # If shapes match, do standard blending
        if pred1.shape == pred2.shape:
            # Initialize with a copy of pred1
            blended = pred1.copy()
            
            # Find cells where predictions differ
            different_cells = (pred1 != pred2)
            
            # If no differences, return original
            if not np.any(different_cells):
                return blended
            
            # For cells that differ, use the prediction from the module with higher weight
            if weight_other > weight_self:
                blended[different_cells] = pred2[different_cells]
            
            return blended
        
        # Handle different shapes
        # Decide which shape to use based on weights
        if weight_other > weight_self:
            # Use shape of pred2 (other prediction)
            target_shape = pred2.shape
            base_pred = pred2.copy()
            other_pred = pred1
        else:
            # Use shape of pred1 (self prediction)
            target_shape = pred1.shape
            base_pred = pred1.copy()
            other_pred = pred2
        
        # Calculate overlapping region
        overlap_rows = min(pred1.shape[0], pred2.shape[0])
        overlap_cols = min(pred1.shape[1], pred2.shape[1])
        
        # Only blend the overlapping region
        for r in range(overlap_rows):
            for c in range(overlap_cols):
                # If cells differ and we need to consider both predictions
                if pred1[r, c] != pred2[r, c]:
                    if weight_self > weight_other:
                        # Keep base_pred (already done since we copied it)
                        pass
                    elif weight_other > weight_self:
                        # Already using other as base, no change needed
                        pass
                    else:
                        # Equal weights, choose randomly 
                        if np.random.random() > 0.5:
                            base_pred[r, c] = other_pred[r, c]
        
        return base_pred
    def _prepare_training_data(self, task, batch_size=8, num_workers=0):
        """
        Prepare training data with rich graph features
        
        Args:
            task: Task object
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            
        Returns:
            DataLoader with preprocessed graphs
        """
        dataloader = self.prepare_gpu_dataset(task, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        
        # Pre-process all graphs
        expected_dim = 3  # or change to 1 if you want only class labels

        for graph in dataloader.dataset.data_items:
            if hasattr(graph, 'x'):
                # Convert from one-hot to class labels if needed
                if graph.x.dim() == 2 and graph.x.size(1) == 11:
                    graph.x = graph.x.argmax(dim=1)

                # Ensure x is long and 2D
                graph.x = graph.x.long()
                if graph.x.dim() == 1:
                    graph.x = graph.x.unsqueeze(1)  # Shape: (900, 1)

                # Standardize shape to (900, expected_dim)
                if graph.x.size(1) < expected_dim:
                    pad = torch.full((graph.x.size(0), expected_dim), self.PADDING_VALUE, dtype=torch.long)
                    pad[:, :graph.x.size(1)] = graph.x
                    graph.x = pad
                elif graph.x.size(1) > expected_dim:
                    graph.x = graph.x[:, :expected_dim]

                # Extract positional info
                graph.pos = graph.x[:, 1:3].float() if expected_dim >= 3 else None
                graph.x = graph.x[:, 0].long().unsqueeze(1)  # Final x: shape (900, 1)

        return dataloader

    def _training_step(self, batch, optimizer, node_loss_fn, edge_loss_fn=None, update_weights=True, **kwargs):
        """
        Perform a training step with both node and edge transformation supervision
        
        Args:
            batch: Training batch
            optimizer: PyTorch optimizer
            node_loss_fn: Loss function for node classification
            edge_loss_fn: Optional loss function for edge prediction
            **kwargs: Additional training parameters
            
        Returns:
            Tuple of (total_loss, node_loss, edge_trans_loss, grid_loss, correct, total, 
                    grid_correct, grid_total, edge_trans_correct, edge_trans_total, 
                    edge_type_correct, edge_type_total)
        """
        # Zero gradients
        optimizer.zero_grad()
        
        # Handle empty or invalid batch
        if batch is None or not hasattr(batch, 'x'):
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        
        # Forward pass
        self.model.train()
        output_batch = self.model(batch)
        
        # Get predictions and targets
        predictions = output_batch.x
        targets = batch.y
        
        # Ensure correct shapes
        if predictions.dim() != 2:
            predictions = predictions.view(-1, self.output_dim)
        
        if targets.dim() > 1:
            targets = targets.view(-1)
        
        # Calculate metrics
        pred_labels = predictions.argmax(dim=1)
        valid_mask = targets != self.PADDING_VALUE  # Ignore padding
        
        # Compute node classification loss
        node_loss = node_loss_fn(predictions[valid_mask], targets[valid_mask])
        total_loss = node_loss
        
        # Compute accuracy
        correct = (pred_labels[valid_mask] == targets[valid_mask]).sum().item()
        total = valid_mask.sum().item()
        
        # Process shape predictions based on padding vs non-padding classification
        shape_correct = 0
        shape_total = 0
        shape_loss = 0.0
        try:
            # Shape prediction: classify nodes as padding (10) vs non-padding (not 10)
            # Convert predictions and targets to binary classification
            pred_is_padding = (pred_labels == self.PADDING_VALUE).float()  # 1 if predicted as padding, 0 otherwise
            target_is_padding = (targets == self.PADDING_VALUE).float()    # 1 if target is padding, 0 otherwise

            # Only consider valid nodes (exclude already padded nodes in input)
            valid_mask = targets != self.PADDING_VALUE
            if valid_mask.sum() > 0:
                # Compute binary cross-entropy loss for shape prediction
                shape_loss = F.binary_cross_entropy(pred_is_padding[valid_mask], target_is_padding[valid_mask])
                total_loss += 0.2 * shape_loss

                # Compute shape accuracy (correct padding/non-padding classification)
                shape_correct = (pred_is_padding[valid_mask].round() == target_is_padding[valid_mask]).sum().item()
                shape_total = valid_mask.sum().item()
            else:
                # If no valid nodes, use all nodes
                shape_loss = F.binary_cross_entropy(pred_is_padding, target_is_padding)
                total_loss += 0.2 * shape_loss

                shape_correct = (pred_is_padding.round() == target_is_padding).sum().item()
                shape_total = targets.size(0)
                
        except Exception as e:
            print(f"Error computing shape loss: {e}")
            shape_loss = 0.0
            # Keep shape metrics at 0 if there was an error
            shape_correct = 0
            shape_total = 0
            
        except Exception as e:
            print(f"Error computing shape loss: {e}")
            import traceback
            traceback.print_exc()
            shape_loss = 0.0
            shape_correct = 0
            shape_total = 0

        # Initialize edge metrics
        edge_trans_correct = 0
        edge_trans_total = 0
        edge_type_correct = 0
        edge_type_total = 0
        edge_trans_loss = 0.0
        edge_type_loss = 0.0
        
        # Process edge transformation prediction
        if hasattr(output_batch, "edge_transformation_pred") and hasattr(output_batch, "edge_transformation_labels"):
            edge_trans_pred = output_batch.edge_transformation_pred
            edge_trans_labels = output_batch.edge_transformation_labels

            # print("[DEBUG] edge_trans_pred:", edge_trans_pred.size(0))
            # print("[DEBUG] edge_trans_labels:", edge_trans_labels.size(0))

            if edge_trans_pred.size(0) != edge_trans_labels.size(0):
                # Try to align by slicing
                min_len = min(edge_trans_pred.size(0), edge_trans_labels.size(0))
                edge_trans_pred = edge_trans_pred[:min_len]
                edge_trans_labels = edge_trans_labels[:min_len]

            if edge_trans_pred.size(0) == edge_trans_labels.size(0):
                edge_trans_loss = F.cross_entropy(edge_trans_pred, edge_trans_labels)
                total_loss += 0.3 * edge_trans_loss
                edge_trans_preds = edge_trans_pred.argmax(dim=1)
                edge_trans_correct = (edge_trans_preds == edge_trans_labels).sum().item()
                edge_trans_total = edge_trans_labels.size(0)
            else:
                print("[SKIP] Edge transformation loss still mismatched.")
            
            if edge_trans_pred.numel() > 0 and edge_trans_labels.numel() > 0:
                # Ensure label shapes match predictions

                # if edge_trans_pred.size(0) != edge_trans_labels.size(0):
                #     print("[SKIP] edge_trans_pred:", edge_trans_pred.size(0))
                #     print("[SKIP] edge_trans_labels:", edge_trans_labels.size(0))
                # else:
                #     print("[OK] edge_trans shapes match:", edge_trans_pred.size())
                if edge_trans_pred.size(0) == edge_trans_labels.size(0):
                    # Compute edge transformation classification loss
                    edge_trans_loss = F.cross_entropy(edge_trans_pred, edge_trans_labels)
                    
                    # Add to total loss with weight
                    total_loss = total_loss + 0.3 * edge_trans_loss
                    
                    # Calculate edge transformation accuracy
                    edge_trans_preds = edge_trans_pred.argmax(dim=1)
                    edge_trans_correct = (edge_trans_preds == edge_trans_labels).sum().item()
                    edge_trans_total = edge_trans_labels.size(0)
        # print("[DEBUG] has edge_type_pred:", hasattr(output_batch, "edge_type_pred"))
        # print("[DEBUG] has edge_type_labels:", hasattr(output_batch, "edge_type_labels"))

        # Process multi-class edge type prediction
        if hasattr(output_batch, "edge_type_pred") and hasattr(output_batch, "edge_type_labels"):
            edge_type_pred = output_batch.edge_type_pred
            edge_type_labels = output_batch.edge_type_labels

            # print("[DEBUG] edge_type_pred shape:", edge_type_pred.shape)
            # print("[DEBUG] edge_type_labels shape:", edge_type_labels.shape)
            # print("[DEBUG] edge_type_labels min/max:", edge_type_labels.min().item(), edge_type_labels.max().item())
            if hasattr(output_batch, "edge_type_pred") and hasattr(output_batch, "edge_type_labels"):
                edge_type_pred = output_batch.edge_type_pred
                edge_type_labels = output_batch.edge_type_labels

                if edge_type_pred.size(0) == edge_type_labels.size(0):
                    edge_type_loss = F.cross_entropy(edge_type_pred, edge_type_labels)
                    total_loss += 0.2 * edge_type_loss

                    edge_type_preds = edge_type_pred.argmax(dim=1)
                    edge_type_correct = (edge_type_preds == edge_type_labels).sum().item()
                    edge_type_total = edge_type_labels.size(0)
                else:
                    print(f"[SKIP] Edge type prediction size mismatch: pred={edge_type_pred.size(0)}, labels={edge_type_labels.size(0)}")
        
            # Calculate grid-level loss
            grid_loss = 0.0
            grid_correct = 0
            grid_total = 0

            # Group by graph for grid-level metrics
            if hasattr(batch, 'batch'):
                num_graphs = batch.batch.max().item() + 1 if batch.batch.numel() > 0 else 1
                grid_total = num_graphs
                
                for graph_idx in range(num_graphs):
                    # Extract predictions and targets for this graph
                    graph_mask = batch.batch == graph_idx
                    graph_preds = predictions[graph_mask]
                    graph_targets = targets[graph_mask]
                    graph_valid = graph_targets != self.PADDING_VALUE
                    
                    if graph_valid.sum() > 0:
                        # Check shape correctness using padding classification
                        graph_pred_labels = graph_preds.argmax(dim=1)

                        # Shape is correct if padding/non-padding classification is correct
                        pred_is_padding = (graph_pred_labels == self.PADDING_VALUE)
                        target_is_padding = (graph_targets == self.PADDING_VALUE)
                        shape_is_correct = (pred_is_padding == target_is_padding).all().item()

                        # Now check node values for non-padding nodes only
                        non_padding_mask = graph_targets != self.PADDING_VALUE
                        if non_padding_mask.sum() > 0:
                            nodes_correct = (graph_pred_labels[non_padding_mask] == graph_targets[non_padding_mask]).all().item()
                        else:
                            nodes_correct = True  # No non-padding nodes to check

                        # Grid is correct only if both shape and nodes are correct
                        is_correct = shape_is_correct and nodes_correct
                        grid_correct += int(is_correct)
                        
                        # Binary grid loss: 1 if not correct, 0 if correct
                        if not is_correct:
                            grid_loss += 1.0 / num_graphs  # Normalize by number of graphs
                
                # Add grid loss to total loss with appropriate weighting
                if grid_total > 0:
                    grid_loss_weight = 0.1
                    grid_loss_tensor = torch.tensor(grid_loss, device=predictions.device, requires_grad=True)
                    total_loss = total_loss + grid_loss_tensor * grid_loss_weight
        
        # Update weights if required
        if update_weights:
            # Backward pass
            total_loss.backward()
            optimizer.step()
        
        return (
            total_loss.item(),
            node_loss.item(),
            edge_trans_loss if isinstance(edge_trans_loss, float) else edge_trans_loss.item(),
            grid_loss.item() if isinstance(grid_loss, torch.Tensor) else grid_loss,
            shape_loss if isinstance(shape_loss, float) else shape_loss.item(),
            shape_correct, shape_total,
            correct, total,
            grid_correct, grid_total,
            edge_trans_correct, edge_trans_total,
            edge_type_correct, edge_type_total,
            edge_type_loss if isinstance(edge_type_loss, float) else edge_type_loss.item()
        )