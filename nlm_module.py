from base_reasoning_module import BaseReasoningModule

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import traceback

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, BatchNorm
from torch_scatter import scatter_mean

class LogicLayer(nn.Module):
    """
    Logic Layer for transformations between predicates of different arities.
    Simplified to focus on unary and binary predicates for stability.
    """
    def __init__(self, in_channels, out_channels, arity_in, arity_out, use_logical_ops=False):
        super(LogicLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.arity_in = arity_in
        self.arity_out = arity_out
        self.use_logical_ops = use_logical_ops
        self.edge_types = ["edge_index", "value_edge_index", "region_edge_index", 
                  "contextual_edge_index", "alignment_edge_index"]
        self.edge_projections = nn.ModuleDict()
        for edge_type in self.edge_types:
            self.edge_projections[edge_type] = nn.Sequential(
                nn.Linear(in_channels * 2, out_channels),
                nn.ReLU()
            )
        self.edge_attention = nn.Parameter(torch.ones(len(self.edge_types)) / len(self.edge_types))
        
        # Set weights for combining logical operations with neural transformations
        if use_logical_ops:
            self.logical_weight = nn.Parameter(torch.tensor(0.5))
            self.neural_weight = nn.Parameter(torch.tensor(0.5))
        
        # For transformations between predicates of same arity
        if arity_in == arity_out:
            self.transform = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU()
            )
        
        # For binary to unary transformation (projection)
        elif arity_in == 2 and arity_out == 1:
            self.projection = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU()
            )

    def logical_and(self, pred1, pred2):
        """Apply logical AND between two predicates (element-wise minimum)"""
        pred2_detached = pred2.detach() if isinstance(pred2, torch.Tensor) else pred2
        return torch.min(pred1, pred2_detached)

    def logical_or(self, pred1, pred2):
        """Apply logical OR between two predicates (element-wise maximum)"""
        pred2_detached = pred2.detach() if isinstance(pred2, torch.Tensor) else pred2
        return torch.maximum(pred1.clone(), pred2_detached)

    def forward(self, x, context=None):
        """
        Forward pass for logic transformations
        
        Args:
            x: Tensor with predicate values
            context: Optional context for transformations
            
        Returns:
            Transformed predicates
        """
        # Handle single value
        if isinstance(x, (int, float)):
            x = torch.tensor([[x]], dtype=torch.float32)
            
        # Handle 1D tensor
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Same arity transformation
        if self.arity_in == self.arity_out:
            transformed = self.transform(x)
            
            # Add logical operations if context is provided
            if self.use_logical_ops and context is not None:
                if context.dim() == 1:
                    context = context.unsqueeze(0)
                    
                # Only apply if dimensions match
                if x.size() == context.size():
                    # Detach context to avoid gradient issues
                    context_detached = context.detach()
                    
                    # Apply logical AND and OR
                    logical_and = self.logical_and(x, context_detached)
                    logical_or = self.logical_or(x, context_detached)
                    
                    # Use learnable weights if available
                    if hasattr(self, 'logical_weight'):
                        logical_weight = torch.sigmoid(self.logical_weight)
                        logical_result = logical_weight * logical_and + (1 - logical_weight) * logical_or
                        
                        neural_weight = torch.sigmoid(self.neural_weight)
                        result = neural_weight * transformed + (1 - neural_weight) * logical_result
                        return result
                    else:
                        # Simple average
                        logical_result = (logical_and + logical_or) / 2
                        return (transformed + logical_result) / 2
            
            return transformed
        
        # Binary to unary transformation (projection)
        elif self.arity_in == 2 and self.arity_out == 1:
            # Handle different input shapes
            if x.dim() == 3:  # batched binary predicates [batch, num_pairs, features]
                batch_size, num_pairs, feature_dim = x.shape
                
                # Reshape to 2D for processing
                x_reshaped = x.view(-1, feature_dim)
                
                # Apply projection
                transformed = self.projection(x_reshaped)
                
                # Reshape back to batched format
                return transformed.view(batch_size, num_pairs, -1)
                
            elif x.dim() == 2:  # standard binary predicates [num_pairs, features]
                # Apply projection directly
                return self.projection(x)
                
            else:
                # Handle unexpected dimensions
                raise ValueError(f"Unexpected input shape for binary-to-unary: {x.shape}")
            
        if hasattr(self, 'edge_projections') and len(self.edge_types) > 1 and context is not None:
            edge_projections = []
            edge_weights = F.softmax(self.edge_attention, dim=0)
            
            for i, edge_type in enumerate(self.edge_types):
                if hasattr(context, edge_type) and getattr(context, edge_type).size(1) > 0:
                    # Get edge index for this type
                    edge_index = getattr(context, edge_type)
                    
                    # Ensure edge_index is not empty
                    if edge_index.size(1) > 0:
                        # Extract features for source and destination nodes
                        src, dst = edge_index
                        edge_features = torch.cat([x[src], x[dst]], dim=1)
                        
                        # Apply edge type projection
                        projected = self.edge_projections[edge_type](edge_features)
                        
                        # Aggregate back to nodes (using mean)
                        node_projection = torch.zeros_like(x)
                        for j in range(len(src)):
                            node_projection[src[j]] += projected[j]
                        
                        # Average by node degree
                        node_counts = torch.bincount(src.cpu(), minlength=x.size(0)).to(x.device)
                        node_counts = node_counts.float().clamp(min=1.0).unsqueeze(1)
                        node_projection = node_projection / node_counts
                        
                        # Apply edge type weight
                        edge_projections.append(node_projection * edge_weights[i])
            
            # Combine edge projections if any
            if edge_projections:
                edge_projection = torch.stack(edge_projections).sum(dim=0)
                # Mix with transformed output if it exists
                if 'transformed' in locals():
                    return transformed * 0.7 + edge_projection * 0.3
                return edge_projection
        
        # Default: return input unchanged
        return x

class EdgeTransformationPredicates:
    """
    Manages logical predicates for edge transformations,
    tracking patterns and confidence scores.
    """
    def __init__(self, confidence_threshold=0.6):
        # Store transformation rules by edge type
        self.transformation_rules = {}
        self.confidence_threshold = confidence_threshold
        self.num_observations = 0
        
    def add_observation(self, edge_type, src_val, dst_val, 
                        new_edge_type=None, exists_in_output=True):
        """
        Add an observation of edge transformation.
        
        Args:
            edge_type: Input edge type
            src_val: Source node value
            dst_val: Destination node value
            new_edge_type: Type that edge transforms to (None if edge disappears)
            exists_in_output: Whether edge exists in output graph
        """
        # Create rule key based on input values
        rule_key = f"{src_val}_{dst_val}"
        
        # Initialize edge type dict if not exists
        if edge_type not in self.transformation_rules:
            self.transformation_rules[edge_type] = {}
            
        # Initialize rule if not exists
        if rule_key not in self.transformation_rules[edge_type]:
            self.transformation_rules[edge_type][rule_key] = {
                "src_val": src_val,
                "dst_val": dst_val,
                "transformations": {},
                "total_observations": 0
            }
        
        # Determine transformation type
        transform_type = "removed" if not exists_in_output else \
                        ("changed" if new_edge_type and new_edge_type != edge_type else "unchanged")
        
        # Get transformation entry, creating if needed
        transformations = self.transformation_rules[edge_type][rule_key]["transformations"]
        if transform_type not in transformations:
            transformations[transform_type] = {
                "count": 0,
                "confidence": 0.0
            }
            
        if transform_type == "changed" and new_edge_type:
            if "target_type" not in transformations[transform_type]:
                transformations[transform_type]["target_type"] = {}
                
            if new_edge_type not in transformations[transform_type]["target_type"]:
                transformations[transform_type]["target_type"][new_edge_type] = {
                    "count": 0,
                    "confidence": 0.0
                }
                
            # Increment count for this specific target type
            transformations[transform_type]["target_type"][new_edge_type]["count"] += 1
            
            # Update confidence for this target type
            type_count = transformations[transform_type]["target_type"][new_edge_type]["count"]
            rule_total = self.transformation_rules[edge_type][rule_key]["total_observations"] + 1
            transformations[transform_type]["target_type"][new_edge_type]["confidence"] = type_count / rule_total
        
        # Increment counts
        transformations[transform_type]["count"] += 1
        self.transformation_rules[edge_type][rule_key]["total_observations"] += 1
        self.num_observations += 1
        
        # Update confidence scores
        self._update_confidences(edge_type, rule_key)
    
    def _update_confidences(self, edge_type, rule_key):
        """Update confidence scores for a rule"""
        rule = self.transformation_rules[edge_type][rule_key]
        total = rule["total_observations"]
        
        # Update confidence for each transformation type
        for transform_type, transform_data in rule["transformations"].items():
            transform_data["confidence"] = transform_data["count"] / total
    
    def get_predicted_transformation(self, edge_type, src_val, dst_val):
        """
        Get predicted transformation for an edge.
        
        Args:
            edge_type: Input edge type
            src_val: Source node value
            dst_val: Destination node value
            
        Returns:
            Dictionary with predicted transformation and confidence
        """
        # Check if we have rules for this edge type
        if edge_type not in self.transformation_rules:
            return {"type": "unchanged", "confidence": 1.0}
            
        # Check if we have a rule for this src-dst pair
        rule_key = f"{src_val}_{dst_val}"
        if rule_key not in self.transformation_rules[edge_type]:
            return {"type": "unchanged", "confidence": 1.0}
            
        # Get transformation with highest confidence
        transformations = self.transformation_rules[edge_type][rule_key]["transformations"]
        if not transformations:
            return {"type": "unchanged", "confidence": 1.0}
            
        # Find highest confidence transformation
        best_type = max(transformations.items(), key=lambda x: x[1]["confidence"])
        transform_type, transform_data = best_type
        
        # Only return transformations with sufficient confidence
        if transform_data["confidence"] < self.confidence_threshold:
            return {"type": "unchanged", "confidence": 1.0}
            
        # For type changes, get the most likely target type
        if transform_type == "changed" and "target_type" in transform_data:
            target_types = transform_data["target_type"]
            if target_types:
                best_target = max(target_types.items(), key=lambda x: x[1]["confidence"])
                target_type, target_data = best_target
                
                if target_data["confidence"] >= self.confidence_threshold:
                    return {
                        "type": transform_type,
                        "target_type": target_type,
                        "confidence": target_data["confidence"]
                    }
        
        # Return the transformation information
        return {
            "type": transform_type,
            "confidence": transform_data["confidence"]
        }
    
    def extract_meta_rules(self):
        """
        Extract higher-level meta-rules from observed transformations.
        These capture general patterns across different node values.
        
        Returns:
            Dictionary of meta-rules
        """
        meta_rules = {}
        
        # Process each edge type
        for edge_type, rules in self.transformation_rules.items():
            meta_rules[edge_type] = {}
            
            # Group transformations by result
            for rule_key, rule_data in rules.items():
                for transform_type, transform_data in rule_data["transformations"].items():
                    # Skip low confidence rules
                    if transform_data["confidence"] < self.confidence_threshold:
                        continue
                        
                    # Create meta-rule key based on transformation type
                    meta_key = transform_type
                    if transform_type == "changed" and "target_type" in transform_data:
                        for target_type, target_data in transform_data["target_type"].items():
                            if target_data["confidence"] >= self.confidence_threshold:
                                sub_key = f"{meta_key}_{target_type}"
                                if sub_key not in meta_rules[edge_type]:
                                    meta_rules[edge_type][sub_key] = {
                                        "count": 0, 
                                        "examples": [],
                                        "confidence": 0.0
                                    }
                                    
                                # Add this rule as an example
                                meta_rules[edge_type][sub_key]["count"] += 1
                                meta_rules[edge_type][sub_key]["examples"].append({
                                    "src_val": rule_data["src_val"],
                                    "dst_val": rule_data["dst_val"],
                                    "confidence": target_data["confidence"]
                                })
                    else:
                        if meta_key not in meta_rules[edge_type]:
                            meta_rules[edge_type][meta_key] = {
                                "count": 0, 
                                "examples": [],
                                "confidence": 0.0
                            }
                            
                        # Add this rule as an example
                        meta_rules[edge_type][meta_key]["count"] += 1
                        meta_rules[edge_type][meta_key]["examples"].append({
                            "src_val": rule_data["src_val"],
                            "dst_val": rule_data["dst_val"],
                            "confidence": transform_data["confidence"]
                        })
            
            # Calculate meta-rule confidences
            for meta_key, meta_data in meta_rules[edge_type].items():
                if meta_data["examples"]:
                    # Average confidence of examples
                    avg_confidence = sum(ex["confidence"] for ex in meta_data["examples"]) / len(meta_data["examples"])
                    meta_data["confidence"] = avg_confidence
        
        return meta_rules
    
class NeuralLogicMachine(nn.Module):
    """
    Neural Logic Machine with Graph Attention Network (GAT) layers
    and improved edge transformation prediction
    """
    def __init__(self, 
            input_dim=3,
            hidden_dim=128, 
            output_dim=11, 
            num_layers=3,
            num_attention_heads=4,
            num_edge_transformation_types=3,
            edge_input_dim=22):
        super(NeuralLogicMachine, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.edge_input_dim = edge_input_dim
        self.num_edge_transformation_types = num_edge_transformation_types
        
        # Initial embedding for node attributes
        self.node_embedding = nn.Embedding(input_dim, hidden_dim)
        self.position_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Add batch normalization for stability
        self.bn_input = nn.BatchNorm1d(hidden_dim)
        
        # Replace GCN layers with GAT layers
        self.gnn_layers = nn.ModuleList()
        self.gnn_bns = nn.ModuleList()
        
        for _ in range(num_layers):
            # GAT layer with multiple attention heads
            gat_layer = GATConv(
                in_channels=hidden_dim, 
                out_channels=hidden_dim // num_attention_heads, 
                heads=num_attention_heads,
                concat=True  # Concatenate multiple attention heads
            )
            self.gnn_layers.append(gat_layer)
            
            # Batch normalization for each layer
            self.gnn_bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Add feature conversion layers for edge processing
        self.node_to_edge_feature = nn.Linear(hidden_dim, 11)
        
        # Define edge type names for consistency
        self.edge_type_names = [
            'edge_index',  # All edge types
            'value_edge_index',
            'region_edge_index', 
            'contextual_edge_index', 
            'alignment_edge_index'
        ]
        
        # Edge type embeddings to capture semantic differences
        self.edge_type_embedding = nn.Embedding(len(self.edge_type_names), hidden_dim // 4)
        
        # Attention mechanism for different edge types - now uses derived features
        self.edge_type_attention = nn.ModuleDict({
            edge_type: nn.Sequential(
                nn.Linear(edge_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            for edge_type in self.edge_type_names[1:]  # Skip spatial edges
        })
        
        # Logic layers for transformations
        self.logic_layers = nn.ModuleList()
        for _ in range(num_layers):
            # Unary to unary transformation (with logical ops)
            self.logic_layers.append(
                LogicLayer(hidden_dim, hidden_dim, 1, 1, use_logical_ops=True)
            )
            
            # Binary to unary transformation
            self.logic_layers.append(
                LogicLayer(hidden_dim * 2, hidden_dim, 2, 1, use_logical_ops=False)
            )
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Output prediction head
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
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
            nn.Linear(hidden_dim, num_edge_transformation_types)
        )

        # Shape predictor
        self.shape_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 4)  # [height_ratio, width_ratio, height_offset, width_offset]
        )

        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights for better training stability"""
        for name, param in self.named_parameters():
            if 'weight' in name and 'bn' not in name and 'batch_norm' not in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def _process_binary_predicates(self, h, edge_index, max_edges=10000):
        """
        Process binary predicates from edges
        
        Args:
            h: Node features tensor [num_nodes, feature_dim]
            edge_index: Edge index tensor [2, num_edges]
            max_edges: Maximum number of edges to consider
            
        Returns:
            Tensor containing binary predicates [num_edges, feature_dim*2]
        """
        if edge_index.size(1) == 0:
            # Return empty tensor with proper shape
            return torch.zeros(0, h.size(1) * 2, device=h.device)
            
        # Sample edges if too many
        if edge_index.size(1) > max_edges and max_edges > 0:
            perm = torch.randperm(edge_index.size(1), device=edge_index.device)[:max_edges]
            edge_index = edge_index[:, perm]
        
        # Fetch source and destination node features
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        # Check indices are within bounds
        valid_src = (src_nodes >= 0) & (src_nodes < h.size(0))
        valid_dst = (dst_nodes >= 0) & (dst_nodes < h.size(0))
        valid_edges = valid_src & valid_dst
        
        if not valid_edges.all():
            # Filter out invalid edges
            src_nodes = src_nodes[valid_edges]
            dst_nodes = dst_nodes[valid_edges]
        
        # Get features for valid edges
        src_features = h[src_nodes]
        dst_features = h[dst_nodes]
        
        # Concatenate features
        binary_preds = torch.cat([src_features, dst_features], dim=1)
        
        return binary_preds
    
    def _process_edge_types(self, h, data):
        """
        Process different edge types with attention mechanism
        
        Args:
            h: Current node features [num_nodes, hidden_dim]
            data: Input graph data
            
        Returns:
            Aggregated edge type features or None
        """
        edge_type_features = []
        
        # Convert hidden dimension features to edge-compatible features
        edge_h = self.node_to_edge_feature(h)  # Convert from hidden_dim to 11
        
        for edge_type in self.edge_type_names[1:]:  # Skip spatial edges (at index 0)
            if hasattr(data, edge_type):
                edge_index = getattr(data, edge_type)
                
                if edge_index.size(1) > 0:  # Check if edge_index is not empty
                    # Extract source and destination node features
                    src, dst = edge_index[0], edge_index[1]
                    
                    # Ensure indices are within bounds
                    valid_src = (src >= 0) & (src < edge_h.size(0))
                    valid_dst = (dst >= 0) & (dst < edge_h.size(0))
                    valid_edges = valid_src & valid_dst
                    
                    if valid_edges.sum() > 0:  # Check if we have valid edges
                        # Filter valid edges
                        src = src[valid_edges]
                        dst = dst[valid_edges]
                        
                        # Use the converted 11-dim features
                        src_features = edge_h[src]
                        dst_features = edge_h[dst]
                        
                        # Concatenate source and destination features
                        edge_features = torch.cat([src_features, dst_features], dim=1)
                        
                        # Compute attention weights for this edge type
                        attention_weights = torch.sigmoid(
                            self.edge_type_attention[edge_type](edge_features)
                        )
                        
                        # Weighted aggregation of edge features
                        aggregated_edge_feature = (src_features * attention_weights).mean(dim=0)
                        aggregated_edge_feature = torch.cat([aggregated_edge_feature, torch.zeros_like(aggregated_edge_feature)])
                        edge_type_features.append(aggregated_edge_feature)
        
        # Combine edge type features if any exist
        if edge_type_features:
            combined = torch.stack(edge_type_features).mean(dim=0)
            # Expand to hidden_dim
            return combined[:h.size(1)] if combined.size(0) >= h.size(1) else F.pad(combined, (0, h.size(1) - combined.size(0)))
        return None
    
    def forward(self, data):
        """
        Forward pass with enhanced graph attention and edge prediction
        
        Args:
            data: PyTorch Geometric data object
            
        Returns:
            Updated data object with processed features and edge predictions
        """
        # Extract batch information
        device = data.x.device
        
        # Initial node embedding
        if data.x.dim() > 1 and data.x.size(1) >= 3:
            # Extract node values and positions
            node_values = data.x[:, 0].long().clamp(0, self.input_dim-1)
            positions = data.x[:, 1:3].float()
            
            # Encode separately
            value_embedding = self.node_embedding(node_values)
            position_encoding = self.position_encoder(positions)
            
            # Combine
            h = torch.cat([value_embedding, position_encoding], dim=1)
        else:
            # Original node value embedding
            node_values = data.x.view(-1).long().clamp(0, self.input_dim-1)
            h = self.node_embedding(node_values)
        
        # Use candidate_edge_index if available (for transformation prediction)
        # Otherwise fall back to regular edge_index
        edge_index = getattr(data, "candidate_edge_index", data.edge_index)
        
        # Keep track of the corresponding edge types
        if hasattr(data, "candidate_edge_type"):
            edge_type = data.candidate_edge_type
        elif hasattr(data, "edge_type"):
            edge_type = data.edge_type
        else:
            # Default to zeros if no edge type information is available
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=device)
        
        # Apply graph attention layers
        for i, (gnn_layer, bn_layer) in enumerate(zip(self.gnn_layers, self.gnn_bns)):
            # Graph Attention Convolution
            h_new = gnn_layer(h, edge_index)
            
            # Batch normalization
            h_new = bn_layer(h_new)
            
            # Apply logic layer transformations
            if i * 2 < len(self.logic_layers):
                unary_layer = self.logic_layers[i * 2]
                
                # Unary transformation
                h_new = unary_layer(h_new)
                
                # Process binary predicates if binary layer exists
                if i * 2 + 1 < len(self.logic_layers):
                    binary_layer = self.logic_layers[i * 2 + 1]
                    binary_preds = self._process_binary_predicates(h_new, edge_index)
                    
                    # Binary to unary projection
                    if binary_preds.size(0) > 0:
                        projected = binary_layer(binary_preds)
                        avg_projection = projected.mean(dim=0, keepdim=True)
                        
                        # Mix with current features
                        h_new = 0.7 * h_new + 0.3 * avg_projection.expand_as(h_new)
            
            # Update features with non-linearity
            h = F.relu(h_new)
            
            # Process edge type features
            edge_type_features = self._process_edge_types(h, data)
            if edge_type_features is not None:
                h = h + edge_type_features
                    
        # Apply output layer for node value prediction
        output = self.output_layer(h)
        
        # Create result
        result = data.clone()
        result.x = output
        result.node_features = h
        
        # Process edge transformation predictions
        if edge_index.numel() > 0:  # Check if there are any edges
            src, dst = edge_index[0], edge_index[1]
            
            # Ensure indices are within bounds
            valid_src = (src >= 0) & (src < h.size(0))
            valid_dst = (dst >= 0) & (dst < h.size(0))
            valid_edges = valid_src & valid_dst
            
            if valid_edges.sum() > 0:
                # Filter valid edges
                filtered_src = src[valid_edges]
                filtered_dst = dst[valid_edges]
                
                # Get node features for edges
                src_features = h[filtered_src]
                dst_features = h[filtered_dst]
                
                # Concatenate for edge features
                edge_features = torch.cat([src_features, dst_features], dim=1)
                
                # Edge transformation prediction if appropriate predictor exists
                if hasattr(self, 'edge_transformation_predictor'):
                    result.edge_transformation_pred = self.edge_transformation_predictor(edge_features)
                
                    # If we have transformation labels, copy them to the result
                    if hasattr(data, "edge_transformation_labels"):
                        # Adjust for valid edges if necessary
                        if valid_edges.sum() < data.edge_transformation_labels.size(0):
                            result.edge_transformation_labels = data.edge_transformation_labels[valid_edges]
                        else:
                            result.edge_transformation_labels = data.edge_transformation_labels
                
                # Edge type prediction if we have the appropriate classifier
                if hasattr(self, 'edge_type_classifier'):
                    result.edge_type_pred = self.edge_type_classifier(edge_features)
                    
                    # Extract the edge type labels that correspond to valid edges
                    filtered_edge_type = edge_type[valid_edges] if valid_edges.sum() < edge_type.size(0) else edge_type
                    
                    # Store these as edge_type_labels for use in training
                    result.edge_type_labels = filtered_edge_type
        
        # Add shape prediction using global graph embedding
        if hasattr(data, 'batch'):
            graph_embedding = scatter_mean(h, data.batch, dim=0)  # [num_graphs, hidden_dim]
        else:
            graph_embedding = h.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        
        # Predict shape transformation
        shape_params = self.shape_predictor(graph_embedding)  # [num_graphs or 1, 4]
        shape_params = shape_params.view(-1, 4)
        result.shape_params = shape_params
        
        return result

    def predict_edge_transformations(self, data):
        """
        Predict edge transformations based on the model's direct predictions.
        
        Args:
            data: Input graph data with edge_transformation_pred from forward pass
            
        Returns:
            Updated graph with predicted edge transformations
        """
        # Create new graph to store predictions
        result = data.clone()
        
        # If the model has already made predictions in the forward pass, use them
        if hasattr(data, "edge_transformation_pred"):
            edge_trans_pred = data.edge_transformation_pred
            
            # Get the edge indices
            edge_index = getattr(data, "candidate_edge_index", data.edge_index)
            
            # Get edge types if available
            edge_types = getattr(data, "candidate_edge_type", getattr(data, "edge_type", None))
            
            if edge_types is None:
                # Default to zeros if no edge type information is available
                edge_types = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
            
            # Get predicted transformation types
            pred_types = edge_trans_pred.argmax(dim=1)
            
            # Confidence values for predictions
            confidences = edge_trans_pred.max(dim=1)[0]
            
            # Threshold for confidence (adjust as needed)
            threshold = 0.7
            
            # Lists to track edges to add, remove, or keep
            edges_to_remove = []
            edges_to_add = {}  # Dictionary mapping edge type to list of edges
            
            # Process each edge
            for i, (pred_type, conf) in enumerate(zip(pred_types, confidences)):
                # Only apply high-confidence predictions
                if conf >= threshold:
                    if pred_type == 1:  # Assuming type 1 means "remove edge"
                        edges_to_remove.append(i)
                    elif pred_type == 2:  # Assuming type 2 means "add edge"
                        # For adding edges, we need the edge type
                        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                        edge_type = edge_types[i].item()
                        
                        # Add to the appropriate edge type
                        if edge_type not in edges_to_add:
                            edges_to_add[edge_type] = []
                        edges_to_add[edge_type].append((src, dst))
                    # Type 0 would be "unchanged", so no action needed
            
            # Remove edges marked for removal
            if edges_to_remove and len(edges_to_remove) < edge_index.size(1):
                keep_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
                keep_mask[torch.tensor(edges_to_remove, device=edge_index.device)] = False
                
                # Update edges and edge types in result
                result.edge_index = edge_index[:, keep_mask]
                if edge_types is not None:
                    result.edge_type = edge_types[keep_mask]
            
            # Add edges to their respective edge types
            # This would depend on how your model expects to handle edge additions
            # For now, we'll just store them in the result for processing
            if edges_to_add:
                result.edges_to_add = edges_to_add
            
            # Store prediction info in the result
            result.edge_transformation_results = {
                "removed": edges_to_remove,
                "added": edges_to_add,
                "num_predictions": edge_trans_pred.size(0)
            }
            
            return result
        else:
            print("No edge transformation predictions available.")
            return data

class NLMReasoningModule(BaseReasoningModule):
    """
    Neural Logic Machine module that integrates with the blackboard architecture
    and supports enhanced edge transformation prediction
    """
    
    def __init__(self, input_dim=11, hidden_dim=128, output_dim=11, 
                 num_layers=3, num_edge_transformation_types=3, device="cpu", use_blackboard_insights=False):
        """Initialize the Neural Logic Machine module"""
        super().__init__()
        
        # Support for grid metrics
        self.supports_grid_metrics = True
        self.use_blackboard_insights = use_blackboard_insights
        
        # Store configuration
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_edge_transformation_types = num_edge_transformation_types
        self.PADDING_VALUE = 10  # Added explicit padding value
        
        # Create the core NLM model
        self.model = self.create_model_from_config({
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'num_layers': num_layers,
            'num_edge_transformation_types': num_edge_transformation_types
        })
        
        # Move to device
        self.device = device
        self.model = self.model.to(self.device)
        
        # Flag for predicate preference
        self.prefers_predicates = True
    
    def create_model_from_config(self, config):
        """
        Create NLM model from configuration with appropriate dimensions
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            New NLM model instance
        """
        # Set edge_input_dim to match feature dimension when concatenating edges
        edge_input_dim = 22  # 11+11 when concatenating two node features
        
        return NeuralLogicMachine(
            input_dim=config.get('input_dim', 11),
            hidden_dim=config.get('hidden_dim', 128),
            output_dim=config.get('output_dim', 11),
            num_layers=config.get('num_layers', 3),
            num_edge_transformation_types=config.get('num_edge_transformation_types', 3),
            edge_input_dim=edge_input_dim
        )
    
    def solve(self, task):
        """
        Solve a task using node prediction and edge transformation prediction 
        with blackboard integration, using pre-computed test graphs.
        
        Args:
            task: Task object
        
        Returns:
            List of predictions for test examples
        """
        # Initialize insights as empty
        insights = {}
        
        # Only retrieve insights if the feature is enabled
        if self.use_blackboard_insights and hasattr(task, 'blackboard') and hasattr(task.blackboard, 'get_insights_for_module'):
            module_name = self.__class__.__name__
            insights = task.blackboard.get_insights_for_module(module_name)
        
        # Access already-learned edge transformation patterns
        transformation_patterns = {}
        try:
            # Just extract meta-rules from the already-learned transformations
            if hasattr(self.model, 'edge_transformations'):
                meta_rules = self.model.edge_transformations.extract_meta_rules()
                transformation_patterns["meta_rules"] = meta_rules
                
                # Store patterns in blackboard if available
                if hasattr(task, 'blackboard'):
                    task.blackboard.update_knowledge(
                        {
                            "edge_transformation_patterns": transformation_patterns,
                            "confidence_threshold": self.model.edge_transformations.confidence_threshold
                        },
                        source='nlm_edge_transformations'
                    )
        except Exception as e:
            print(f"[solve] Could not access edge transformation patterns: {e}")
            traceback.print_exc()
        
        predictions = []
        start_time = time.time()

        try:
            self.model.eval()

            # Process each test example using pre-computed test graphs
            for i, test_graph in enumerate(task.test_graphs):
                # Get original test input for reference
                test_input, _ = task.test_pairs[i]
                
                # Get the input shape for shape prediction reference
                input_shape = np.array(test_input).shape
                
                # Skip preprocessing if already done
                if not hasattr(test_graph, 'preprocessed') or not test_graph.preprocessed:
                    # This code would run only if the graph wasn't preprocessed earlier
                    print(f"Preprocessing test graph {i}...")
                    if hasattr(test_graph, 'x'):
                        # Convert from one-hot to class labels if needed
                        if test_graph.x.dim() == 2 and test_graph.x.size(1) == 11:
                            test_graph.x = test_graph.x.argmax(dim=1)

                        # Ensure x is long and 2D
                        test_graph.x = test_graph.x.long()
                        if test_graph.x.dim() == 1:
                            test_graph.x = test_graph.x.unsqueeze(1)  # Shape: (900, 1)

                        # Standardize shape to (900, expected_dim)
                        expected_dim = 3  # Same as in _prepare_training_data
                        if test_graph.x.size(1) < expected_dim:
                            pad = torch.full((test_graph.x.size(0), expected_dim), self.PADDING_VALUE, dtype=torch.long)
                            pad[:, :test_graph.x.size(1)] = test_graph.x
                            test_graph.x = pad
                        elif test_graph.x.size(1) > expected_dim:
                            test_graph.x = test_graph.x[:, :expected_dim]

                        # Extract positional info
                        test_graph.pos = test_graph.x[:, 1:3].float() if expected_dim >= 3 else None
                        test_graph.x = test_graph.x[:, 0].long().unsqueeze(1)  # Final x: shape (900, 1)
                
                # Move graph to device
                test_graph = test_graph.to(self.device)

                # Apply any predicates from insights if available
                if insights and 'predicates' in insights and insights['predicates']:
                    # Log that we're using predicate patterns
                    print(f"Using {len(insights['predicates'])} predicates from blackboard")

                with torch.no_grad():
                    # For single graphs, add an explicit batch dimension to avoid scatter_mean issues
                    if not hasattr(test_graph, 'batch') or test_graph.batch is None:
                        # Create a batch attribute with all zeros (single graph)
                        test_graph.batch = torch.zeros(test_graph.x.size(0), dtype=torch.long, device=test_graph.x.device)
                    
                    output = self.model(test_graph)

                    # Get the predicted shape from the model
                    predicted_shape = input_shape  # Default to input shape if no prediction available
                    shape_info = None
                    
                    if hasattr(output, 'shape_params'):
                        try:
                            # Get the predicted shape parameters
                            shape_params = output.shape_params.cpu().numpy()[0]  # [height_ratio, width_ratio, height_offset, width_offset]
                            
                            # Calculate new dimensions
                            height_ratio, width_ratio, height_offset, width_offset = shape_params
                            predicted_height = max(1, int(round(input_shape[0] * height_ratio + height_offset)))
                            predicted_width = max(1, int(round(input_shape[1] * width_ratio + width_offset)))
                            predicted_shape = (predicted_height, predicted_width)
                            
                            shape_info = {
                                "predicted_params": shape_params.tolist(),
                                "input_shape": input_shape,
                                "predicted_shape": predicted_shape
                            }
                        except Exception as e:
                            print(f"Error in shape prediction: {e}")
                            predicted_shape = input_shape
                            shape_info = {
                                "error": str(e),
                                "input_shape": input_shape,
                                "predicted_shape": input_shape
                            }

                    # Apply edge transformation predictions if available
                    if hasattr(self.model, 'predict_edge_transformations'):
                        output = self.model.predict_edge_transformations(output)

                    # Convert model output back to grid with predicted shape
                    prediction = task.graph_to_grid(output, predicted_shape)
                
                # Enhance prediction with blackboard insights if available
                if insights and insights.get('high_confidence_predictions'):
                    # Find relevant predictions for this test example
                    for pred_info in insights['high_confidence_predictions']:
                        other_predictions = pred_info['prediction']
                        
                        # Check if this prediction applies to our current example
                        if isinstance(other_predictions, list) and i < len(other_predictions):
                            other_pred = other_predictions[i]
                            
                            # If it's a high-confidence prediction from another module (e.g., LLM)
                            # and confidence is higher than ours, incorporate it
                            if pred_info['confidence'] > 0.85:
                                # For NLM, we might want to be more selective about which cells to update
                                if other_pred.shape == prediction.shape:
                                    # Create a weighted blend based on confidence
                                    prediction = self.enhance_prediction_with_insight(
                                        prediction, other_pred, 
                                        self_confidence=0.8,  # Default confidence for NLM predictions
                                        other_confidence=pred_info['confidence']
                                    )
                                    
                                    # Log that we enhanced the prediction
                                    print(f"Enhanced NLM prediction using insights from {pred_info['module']}")
                
                predictions.append(prediction)

                # Extract edge transformation info for logging
                edge_transformation_info = None
                if hasattr(output, 'edge_transformation_pred'):
                    edge_transformation_info = {
                        "mean_confidence": float(output.edge_transformation_pred.max(dim=1)[0].mean().item()),
                        "num_edges": output.edge_transformation_pred.size(0)
                    }

                # Log each prediction step with shape information
                task.log_reasoning_step(
                    module_name=self.__class__.__name__,
                    details={
                        "action": f"predict_example_{i}_with_node_and_edge_transformation",
                        "shape": prediction.shape,
                        "edge_transformations": edge_transformation_info,
                        "shape_prediction": shape_info,  # Add shape prediction info
                        "used_blackboard_insights": bool(insights and (insights.get('predicates') or 
                                                        insights.get('high_confidence_predictions'))),
                        "used_precomputed_graphs": True
                    }
                )

            inference_time = time.time() - start_time
            if hasattr(self, 'inference_time'):
                self.inference_time.append(inference_time)

            # Get the final confidence from the last output if available
            confidence = 0.5  # Default confidence
            if 'output' in locals() and hasattr(output, 'x'):
                node_logits = output.x
                confidence = float(F.softmax(node_logits, dim=1).max(dim=1)[0].mean().item())
            
            task.log_reasoning_step(
                module_name=self.__class__.__name__,
                prediction=predictions,
                confidence=confidence,
                time_taken=inference_time,
                details={
                    "action": "final_predictions_with_edge_transformations_and_shape_prediction",
                    "used_transformation_patterns": bool(transformation_patterns),
                    "used_blackboard_insights": bool(insights),
                    "used_precomputed_graphs": True
                }
            )

            return predictions

        except Exception as e:
            print(f"Error in combined solve: {e}")
            traceback.print_exc()

            # Fallback: copy inputs as predictions
            for test_input, _ in task.test_pairs:
                predictions.append(np.array(test_input))

            task.log_reasoning_step(
                module_name=self.__class__.__name__,
                prediction=predictions,
                confidence=0.1,
                time_taken=time.time() - start_time,
                details={"error": str(e)}
            )

            return predictions

    def enhance_prediction_with_insight(self, prediction, other_prediction, self_confidence=0.8, other_confidence=0.9):
        """
        Enhance NLM prediction with insight from another module
        
        Args:
            prediction: NLM prediction grid
            other_prediction: Other module prediction grid
            self_confidence: Confidence in NLM prediction
            other_confidence: Confidence in other prediction
            
        Returns:
            Enhanced prediction grid
        """
        # If shapes don't match, return original prediction
        if prediction.shape != other_prediction.shape:
            return prediction
        
        # Initialize with a copy of the original prediction
        enhanced = prediction.copy()
        
        # Find cells where predictions differ
        diff_mask = (prediction != other_prediction)
        
        if not np.any(diff_mask):
            return enhanced  # No differences to resolve
        
        # Calculate confidence ratio for comparison
        confidence_ratio = other_confidence / self_confidence
        
        # If the other module has significantly higher confidence, 
        # selectively incorporate its predictions
        if confidence_ratio > 1.1:  # Other module is at least 10% more confident
            # For cells where predictions differ, adopt the other module's prediction
            # with probability proportional to confidence ratio
            for i, j in zip(*np.where(diff_mask)):
                # Weighted probability based on confidence
                if np.random.random() < (confidence_ratio - 1.0) / confidence_ratio:
                    enhanced[i, j] = other_prediction[i, j]
        
        return enhanced
    
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
        Perform a training step with both node and edge supervision,
        following the unified module approach.

        Args:
            batch: Training batch
            optimizer: PyTorch optimizer
            node_loss_fn: Loss function for node classification
            edge_loss_fn: Loss function for edge classification
            **kwargs: Additional training parameters

        Returns:
            Tuple with full metrics including edge transformation metrics
        """
        if batch is None or not hasattr(batch, 'x'):
            return 0.0, 0.0, 0.0, 0.0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0

        optimizer.zero_grad()
        self.model.train()
        
        try:
            # Forward pass
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


            # Process shape predictions
            shape_correct = 0
            shape_total = 0
            shape_loss = 0.0
            try:
                # Make sure dimensions match
                pred_shape = output_batch.shape_params
                target_shape = batch.shape_params
                
                # Set shape_total to the batch size
                shape_total = pred_shape.size(0)
                
                # Ensure the dimensions match
                if len(pred_shape.shape) == 2 and len(target_shape.shape) == 1:
                    # Reshape target to match prediction
                    batch_size = pred_shape.shape[0]
                    if target_shape.shape[0] == batch_size * 4:
                        target_shape = target_shape.view(batch_size, 4)
                
                # Compute loss
                shape_loss = F.mse_loss(pred_shape, target_shape)
                total_loss += 0.2 * shape_loss
                
                # Check exact shape correctness
                for i in range(pred_shape.size(0)):
                    # Check if shape parameters match exactly
                    is_shape_correct = torch.all(pred_shape[i].round() == target_shape[i].round()).item()
                    shape_correct += int(is_shape_correct)
                    
            except Exception as e:
                print(f"Error computing shape loss: {e}")
                shape_loss = 0.0
                # Keep shape metrics at 0 if there was an error
                shape_correct = 0
                shape_total = 0
                
            # Initialize edge metrics
            edge_trans_correct = 0
            edge_trans_total = 0
            edge_type_correct = 0
            edge_type_total = 0
            edge_trans_loss = 0.0
            edge_type_loss = 0.0
            grid_loss = 0.0

            # Process edge transformation prediction
            if hasattr(output_batch, "edge_transformation_pred") and hasattr(output_batch, "edge_transformation_labels"):
                edge_trans_pred = output_batch.edge_transformation_pred
                edge_trans_labels = output_batch.edge_transformation_labels

                if edge_trans_pred.size(0) == edge_trans_labels.size(0) and edge_trans_pred.size(0) > 0:
                    edge_trans_loss = F.cross_entropy(edge_trans_pred, edge_trans_labels)
                    total_loss += 0.3 * edge_trans_loss
                    edge_trans_preds = edge_trans_pred.argmax(dim=1)
                    edge_trans_correct = (edge_trans_preds == edge_trans_labels).sum().item()
                    edge_trans_total = edge_trans_labels.size(0)
            else:
                print("No edge transformation prediction available.")

            # Process multi-class edge type prediction
            if hasattr(output_batch, "edge_type_pred") and hasattr(output_batch, "edge_type_labels"):
                edge_type_pred = output_batch.edge_type_pred
                edge_type_labels = output_batch.edge_type_labels

                if edge_type_pred.size(0) == edge_type_labels.size(0) and edge_type_pred.size(0) > 0:
                    edge_type_loss = F.cross_entropy(edge_type_pred, edge_type_labels)
                    total_loss += 0.2 * edge_type_loss

                    edge_type_preds = edge_type_pred.argmax(dim=1)
                    edge_type_correct = (edge_type_preds == edge_type_labels).sum().item()
                    edge_type_total = edge_type_labels.size(0)
            else:
                print("No edge type prediction available.")
                # Silently proceed without edge type prediction
                edge_type_loss = 0.0
                edge_type_correct = 0
                edge_type_total = 0

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
                        # First check shape correctness
                        shape_is_correct = False
                        
                        # Check shape if shape parameters are available
                        if hasattr(output_batch, 'shape_params') and hasattr(batch, 'shape_params') and \
                        graph_idx < output_batch.shape_params.size(0) and graph_idx < batch.shape_params.size(0):
                            pred_shape_params = output_batch.shape_params[graph_idx]
                            target_shape_params = batch.shape_params[graph_idx]
                            shape_is_correct = torch.all(pred_shape_params.round() == target_shape_params.round()).item()
                        
                        # Now check node values
                        nodes_correct = (graph_preds[graph_valid].argmax(dim=1) == graph_targets[graph_valid]).all().item()
                        
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
            
        except Exception as e:
            print(f"Error in batch {batch.batch[-1] if hasattr(batch, 'batch') else 0}: {str(e)}")
            # Return default values
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0,0, 1, 0, 1, 0, 0, 0, 0, 0.0