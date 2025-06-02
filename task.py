import torch
import networkx as nx
import torch_geometric
from torch_geometric.data import Data, Batch
import numpy as np
import json
import os
import time
import copy

class Blackboard:
    """
    Blackboard with unified storage format for predicates and transformations.
    Provides efficient representation and querying of grid pattern knowledge.
    """
    def __init__(self, max_grid_size=30*30):
        # Unified knowledge store
        self.knowledge_base = {}
        self.knowledge_sources = {}
        
        # Value table: efficient representation of node values
        # Rows represent values (0-10), columns represent nodes
        self.value_table = np.zeros((11, max_grid_size), dtype=np.uint8)
        
        # Tracking reasoning history
        self.reasoning_history = []
        self.confidence_scores = {}
        
        # Textual data for LLM
        self.textual_data = []
        
        # Graph data for GNN
        self.graph_data = None
        
        # Flag for batched training
        self.is_batched_training = False
        
        # Maximum grid size
        self.max_grid_size = max_grid_size
    
    def update_knowledge(self, knowledge_dict, source='generic'):
        """
        Update knowledge base with unified format information.
        
        Args:
            knowledge_dict: Dictionary of knowledge items 
            source: Source of the knowledge
            
        Returns:
            self for method chaining
        """
        for key, value in knowledge_dict.items():
            self.knowledge_base[key] = value
            self.knowledge_sources[key] = source
            
            # Update value table for node values
            if key.startswith("has_value_"):
                try:
                    # Extract value from key (format: has_value_X)
                    value_id = int(key.split("_")[-1])
                    if isinstance(value, list) and value_id < 11:
                        # Reset this value's row
                        self.value_table[value_id, :] = 0
                        # Set bits for nodes that have this value
                        for node_id in value:
                            if node_id < self.max_grid_size:
                                self.value_table[value_id, node_id] = 1
                except (ValueError, IndexError):
                    pass
        
        return self
    
    def get_insights_for_module(self, module_name):
        """
        Extract insights contributed by other modules.
        
        Args:
            module_name: Name of the requesting module
            
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
        
        # Extract from knowledge base
        if hasattr(self, 'knowledge_base'):
            for key, value in self.knowledge_base.items():
                source = self.knowledge_sources.get(key, "unknown")
                
                # Skip requesting module's own contributions
                if source == module_name:
                    continue
                    
                # Add to appropriate category
                if "transform" in key.lower():
                    insights['transformations'][key] = value
                elif "predicate" in key.lower():
                    insights['predicates'][key] = value
                elif "pattern" in key.lower():
                    insights['patterns'][key] = value
        
        # Extract confidence scores
        if hasattr(self, 'confidence_scores'):
            insights['confidence_scores'] = {
                module: scores for module, scores in self.confidence_scores.items()
                if module != module_name
            }
        
        return insights
    
    def get_nodes_with_value(self, value):
        """
        Get all nodes with a specific value.
        
        Args:
            value: Value to search for (0-10)
            
        Returns:
            List of node indices
        """
        if 0 <= value < 11:
            return np.where(self.value_table[value] == 1)[0].tolist()
        return []
    
    def get_node_value(self, node_id):
        """
        Get the value of a specific node.
        
        Args:
            node_id: Node index
            
        Returns:
            Value of the node (0-10) or None if not found
        """
        if 0 <= node_id < self.max_grid_size:
            values = np.where(self.value_table[:, node_id] == 1)[0]
            return values[0] if len(values) > 0 else None
        return None
    
    def set_node_value(self, node_id, value):
        """
        Set the value for a specific node.
        
        Args:
            node_id: Node index
            value: Value to set (0-10)
            
        Returns:
            self for method chaining
        """
        if 0 <= node_id < self.max_grid_size and 0 <= value < 11:
            # Clear any existing values for this node
            self.value_table[:, node_id] = 0
            # Set the new value
            self.value_table[value, node_id] = 1
            
            # Update knowledge base
            value_key = f"has_value_{value}"
            if value_key not in self.knowledge_base:
                self.knowledge_base[value_key] = []
            
            if node_id not in self.knowledge_base[value_key]:
                self.knowledge_base[value_key].append(node_id)
        
        return self


class Task:
    """
    Enhanced task representation for ARC-AGI tasks with blackboard architecture:
    - Enables communication between GNN, NLM, and LLM reasoning modules
    - Provides unified data conversion between different representations
    - Maintains history of reasoning steps and transformations
    - Supports meta-reasoning through standardized interfaces
    """
    
    GLOBAL_MAX_NODES = 30 * 30  # Set to 900 to match max grid size
    PADDING_VALUE = 10          # Value used for padding

    def __init__(self, task_id, train_pairs, test_pairs):
        self.task_id = task_id
        self.train_pairs = train_pairs
        self.test_pairs = test_pairs
        
        # Initialize blackboard for module communication
        self.blackboard = Blackboard()
        
        # Define edge types for consistency
        self.edge_types = [
            "edge_index", 
            "value_edge_index", 
            "region_edge_index", 
            "contextual_edge_index", 
            "alignment_edge_index"
        ]
        
        # Convert training examples into structured graph representations
        self.train_graphs = []
        for input_grid, output_grid in train_pairs:
            # Create input graph with all edge types
            graph = self.grid_to_graph(
                input_grid,
                logical=True,
                add_value_based_edges=True,
                add_region_edges=True,
                add_contextual_edges=True,
                add_alignment_edges=True
            )
            graph.y = self.pad_target_grid(output_grid).view(-1)

            # Create output graph with the same edge types for comparison
            output_graph = self.grid_to_graph(
                output_grid,
                logical=True,
                add_value_based_edges=True,
                add_region_edges=True,
                add_contextual_edges=True,
                add_alignment_edges=True
            )
            
            # Initialize all edge attributes with empty tensors for consistency
            for edge_type in self.edge_types:
                if not hasattr(graph, edge_type):
                    setattr(graph, edge_type, torch.tensor([[], []], dtype=torch.long))
                if not hasattr(output_graph, edge_type):
                    setattr(output_graph, edge_type, torch.tensor([[], []], dtype=torch.long))
                
                # Initialize edge labels with empty tensors
                setattr(graph, f"{edge_type}_label", torch.tensor([], dtype=torch.long))
                # Initialize transformation labels with empty tensors
                setattr(graph, f"{edge_type}_transformation", torch.tensor([], dtype=torch.long))

            # Extract and apply edge transformations
            graph = self.extract_edge_transformations(graph, output_graph)

            self.train_graphs.append(graph)

        # Convert test examples into structured graph representations
        self.test_graphs = []
        for input_grid, output_grid in test_pairs:
            # Create input graph with all edge types
            graph = self.grid_to_graph(
                input_grid,
                logical=True,
                add_value_based_edges=True,
                add_region_edges=True,
                add_contextual_edges=True,
                add_alignment_edges=True
            )
            graph.y = self.pad_target_grid(output_grid).view(-1)

            # Create output graph with the same edge types for comparison
            output_graph = self.grid_to_graph(
                output_grid,
                logical=True,
                add_value_based_edges=True,
                add_region_edges=True,
                add_contextual_edges=True,
                add_alignment_edges=True
            )
            
            # Initialize all edge attributes with empty tensors for consistency
            for edge_type in self.edge_types:
                if not hasattr(graph, edge_type):
                    setattr(graph, edge_type, torch.tensor([[], []], dtype=torch.long))
                if not hasattr(output_graph, edge_type):
                    setattr(output_graph, edge_type, torch.tensor([[], []], dtype=torch.long))
                
                # Initialize edge labels with empty tensors
                setattr(graph, f"{edge_type}_label", torch.tensor([], dtype=torch.long))
                # Initialize transformation labels with empty tensors
                setattr(graph, f"{edge_type}_transformation", torch.tensor([], dtype=torch.long))

            # Extract and apply edge transformations
            graph = self.extract_edge_transformations(graph, output_graph)

            self.test_graphs.append(graph)

        # Create padded target tensors
        self.train_targets = [self.pad_target_grid(pair[1]) for pair in train_pairs]
        self.test_targets = [self.pad_target_grid(pair[1]) for pair in test_pairs]

        # Track feature extraction methods used
        self.features = {}
    
    def pad_target_grid(self, output_grid):
        """
        Pads output grid to match 30x30 = 900 nodes.
        Uses class `10` as the padding value.
        """
        # Convert to a NumPy array first (to check shape)
        try:
            output_grid = np.array(output_grid, dtype=np.int64)
        except ValueError:
            raise ValueError(f"ERROR: output_grid contains non-uniform row lengths. Fix jagged input!")

        # Ensure the grid is 2D (not jagged)
        if output_grid.ndim != 2:
            raise ValueError(f"ERROR: output_grid must be a 2D array, but got shape {output_grid.shape}")

        # Force output grid to be exactly 30x30
        padded_grid = np.full((30, 30), self.PADDING_VALUE, dtype=np.int64)  # Default pad with class `10`

        # Copy original grid into padded grid
        rows, cols = output_grid.shape
        padded_grid[:rows, :cols] = output_grid

        # Convert to PyTorch tensor and flatten
        return torch.tensor(padded_grid, dtype=torch.long).flatten()

    def grid_to_graph(self, grid, logical=False, add_value_based_edges=True,
                add_region_edges=True, add_contextual_edges=True,
                add_alignment_edges=True, context_window_size=5,
                context_similarity_threshold=0.7):
        """
        Unified function to convert a grid to a graph representation
        with positional features and optional semantic edges.
        
        Args:
            grid: Input grid (2D array)
            logical: Whether to extract logical features for NLM
            add_value_based_edges: Whether to add value-based edges
            add_region_edges: Whether to add edges between cells in the same region
            add_contextual_edges: Whether to add edges between cells with similar local patterns
            add_alignment_edges: Whether to add edges between cells aligned in rows or columns
            context_window_size: Size of the window for contextual patterns (must be odd number)
            context_similarity_threshold: Threshold for considering contexts similar (0.0-1.0)
            
        Returns:
            PyG Data object with appropriate features
        """
        # Pad the grid to standard size (30x30)
        padded_grid = self.pad_target_grid(grid)
        padded_grid = padded_grid.view(30, 30)  # Reshape into 30x30 for processing
        
        # Convert input grid to numpy array
        grid_np = np.array(grid)
        actual_rows, actual_cols = grid_np.shape

        # Helper function to check if a cell is valid (within original grid and not padding)
        def is_valid_cell(r, c):
            return (0 <= r < actual_rows and 
                    0 <= c < actual_cols and 
                    padded_grid[r, c] != self.PADDING_VALUE)

        # Create graph and node mapping
        G = nx.Graph()
        node_positions = {}  # Maps (r,c) to node_id

        # Add nodes with enhanced features
        for r in range(30):
            for c in range(30):
                node_id = r * 30 + c
                
                # Create node features: [value, normalized_row, normalized_col]
                node_features = [
                    float(padded_grid[r, c]),  # Cell value
                    r / 29.0,  # Normalized row position
                    c / 29.0   # Normalized column position
                ]
                
                G.add_node(node_id, features=node_features)
                
                # Store position mapping for semantic edges
                if is_valid_cell(r, c):
                    node_positions[(r, c)] = node_id

                # Add spatial edges for adjacent cells (4-connectivity)
                if r > 0: G.add_edge(node_id, (r - 1) * 30 + c)  # Up
                if c > 0: G.add_edge(node_id, r * 30 + (c - 1))  # Left
        
        # Extract spatial edges
        spatial_edges = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        
        # Initialize semantic edge lists
        value_based_edges, region_edges, contextual_edges, alignment_edges = [], [], [], []
        
        # Value-based edges
        if add_value_based_edges:
            value_groups = {}
            for r in range(actual_rows):
                for c in range(actual_cols):
                    val = grid_np[r, c]
                    if val != 0 and val != self.PADDING_VALUE:
                        if val not in value_groups:
                            value_groups[val] = []
                        value_groups[val].append((r, c))
            
            # Connect same-value cells (not adjacent)
            for val, positions in value_groups.items():
                for i, (r1, c1) in enumerate(positions):
                    for (r2, c2) in positions[i+1:]:
                        if abs(r1-r2) + abs(c1-c2) > 1:
                            node1 = node_positions[(r1, c1)]
                            node2 = node_positions[(r2, c2)]
                            value_based_edges.extend([[node1, node2], [node2, node1]])
        
        # Region edges
        if add_region_edges:
            visited = np.zeros((actual_rows, actual_cols), dtype=bool)
            
            def find_region(r, c, value):
                region = []
                queue = [(r, c)]
                visited[r, c] = True
                
                while queue:
                    curr_r, curr_c = queue.pop(0)
                    region.append((curr_r, curr_c))
                    
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if (0 <= nr < actual_rows and 0 <= nc < actual_cols and 
                            not visited[nr, nc] and grid_np[nr, nc] == value):
                            queue.append((nr, nc))
                            visited[nr, nc] = True
                
                return region
            
            for r in range(actual_rows):
                for c in range(actual_cols):
                    if not visited[r, c] and grid_np[r, c] != 0 and grid_np[r, c] != self.PADDING_VALUE:
                        region = find_region(r, c, grid_np[r, c])
                        
                        # Connect all nodes in the region
                        for i, (r1, c1) in enumerate(region):
                            for (r2, c2) in region[i+1:]:
                                node1 = node_positions[(r1, c1)]
                                node2 = node_positions[(r2, c2)]
                                region_edges.extend([[node1, node2], [node2, node1]])
        
        # Add contextual pattern edges if requested
        if add_contextual_edges:
            # Ensure window size is odd
            if context_window_size % 2 == 0:
                context_window_size += 1
            
            half_window = context_window_size // 2
            
            # Create pattern vectors for each cell
            pattern_vectors = {}
            for r in range(half_window, actual_rows - half_window):
                for c in range(half_window, actual_cols - half_window):
                    # Extract the window around this cell
                    pattern = []
                    for dr in range(-half_window, half_window + 1):
                        for dc in range(-half_window, half_window + 1):
                            pattern.append(grid_np[r + dr, c + dc])
                    
                    pattern_vectors[(r, c)] = np.array(pattern)
            
            # Connect cells with similar contexts (sparse selection)
            contextual_edges_set = set()
            for (r1, c1), pattern1 in pattern_vectors.items():
                similarities = []
                for (r2, c2), pattern2 in pattern_vectors.items():
                    if (r1, c1) != (r2, c2):
                        similarity = np.sum(pattern1 == pattern2) / len(pattern1)
                        if similarity >= context_similarity_threshold:
                            node1 = node_positions[(r1, c1)]
                            node2 = node_positions[(r2, c2)]
                            similarities.append((similarity, node1, node2))

                # Sort and keep top-k similar neighbors per node
                similarities.sort(reverse=True)
                top_k = 5  # tune this based on sparsity needs
                for _, node1, node2 in similarities[:top_k]:
                    if (node1, node2) not in contextual_edges_set:
                        contextual_edges.append([node1, node2])
                        contextual_edges.append([node2, node1])
                        contextual_edges_set.add((node1, node2))
                        contextual_edges_set.add((node2, node1))
        
        # Add alignment edges if requested
        if add_alignment_edges:
            # Row alignment edges (connect non-zero cells in the same row)
            for r in range(actual_rows):
                row_cells = [(r, c) for c in range(actual_cols) if grid_np[r, c] != 0]
                for i, pos1 in enumerate(row_cells):
                    for pos2 in row_cells[i+1:]:
                        node1 = node_positions[pos1]
                        node2 = node_positions[pos2]
                        alignment_edges.append([node1, node2])
                        alignment_edges.append([node2, node1])
            
            # Column alignment edges (connect non-zero cells in the same column)
            for c in range(actual_cols):
                col_cells = [(r, c) for r in range(actual_rows) if grid_np[r, c] != 0]
                for i, pos1 in enumerate(col_cells):
                    for pos2 in col_cells[i+1:]:
                        node1 = node_positions[pos1]
                        node2 = node_positions[pos2]
                        alignment_edges.append([node1, node2])
                        alignment_edges.append([node2, node1])
        
        # Create node features directly from the padded grid
        node_features = []
        for r in range(30):
            for c in range(30):
                val = float(padded_grid[r, c])
                node_features.append([val, r / 29.0, c / 29.0])
        graph = Data(x=torch.tensor(node_features, dtype=torch.float32))

        # Add safe spatial edges
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        valid_nodes = torch.arange(900)
        edge_mask = (edge_index[0] < 900) & (edge_index[1] < 900)
        graph.edge_index = edge_index[:, edge_mask]
        
        # Add various edge types
        edge_types = [
            ('value_edge_index', value_based_edges),
            ('region_edge_index', region_edges),
            ('contextual_edge_index', contextual_edges),
            ('alignment_edge_index', alignment_edges)
        ]
        
        # Create edge type tensors
        edge_type_tensors = []
        type_id = 1  # Start from 1 as 0 is for spatial edges
        
        # Add spatial edges (type 0)
        if spatial_edges.size(1) > 0:
            spatial_edge_type = torch.zeros(spatial_edges.size(1), dtype=torch.long)
            edge_type_tensors.append((spatial_edges, spatial_edge_type))
        
        # Process other edge types
        for edge_name, edges in edge_types:
            if edges:
                edge_tensor = torch.tensor(edges, dtype=torch.long).t()
                setattr(graph, edge_name, edge_tensor)  # Set individually for each edge type
                edge_type_tensors.append((edge_tensor, torch.full((edge_tensor.size(1),), type_id, dtype=torch.long)))
                type_id += 1
        
        # Combine all edge types
        if edge_type_tensors:
            all_edges = torch.cat([et[0] for et in edge_type_tensors], dim=1)
            all_edge_types = torch.cat([et[1] for et in edge_type_tensors])
            
            # Set combined edges and types
            graph.edge_index = all_edges
            graph.edge_type = all_edge_types
        
        # Logical feature processing
        if logical:
            x = graph.x
            values = x[:, 0].long()
            one_hot = torch.zeros(x.size(0), 11)
            one_hot.scatter_(1, values.unsqueeze(1), 1)
            graph.original_features = graph.x.clone()
            graph.x = one_hot
            graph.feature_type = "logical"
        else:
            graph.feature_type = "standard"
        
        # Store graph in blackboard
        if hasattr(self.blackboard, "update_knowledge"):
            self.blackboard.update_knowledge({"graph_data": graph})

        return graph
    
    def extract_edge_transformations(self, input_graph, output_graph):
        """
        Extract edge transformation labels from the union of edges across input/output graphs.
        Stores the new edge set under a separate name so original graph data is untouched.
        """
        # Convert to sets of edge tuples
        E_in = set(map(tuple, input_graph.edge_index.t().tolist()))
        E_out = set(map(tuple, output_graph.edge_index.t().tolist()))

        # Union of edges
        candidate_edges = list(E_in | E_out)
        labels = []

        for edge in candidate_edges:
            if edge in E_in and edge in E_out:
                labels.append(0)  # unchanged
            elif edge in E_in:
                labels.append(1)  # removed
            else:
                labels.append(2)  # added

        # Save as new fields instead of overwriting edge_index
        input_graph.candidate_edge_index = torch.tensor(candidate_edges, dtype=torch.long).t().contiguous()
        input_graph.edge_transformation_labels = torch.tensor(labels, dtype=torch.long)

        # Re-align edge types based on original edge_index
        if hasattr(input_graph, "edge_type") and hasattr(input_graph, "edge_index"):
            edge_to_type = {
                tuple(edge): et for edge, et in zip(
                    input_graph.edge_index.t().tolist(),
                    input_graph.edge_type.tolist()
                )
            }

            new_types = [edge_to_type.get(tuple(edge), 0) for edge in candidate_edges]
            input_graph.candidate_edge_type = torch.tensor(new_types, dtype=torch.long)

        return input_graph
    
    def graph_to_grid(self, graph_data, output_shape=None):
        """
        Convert graph data back to grid format, using predicted shape if available
        
        Args:
            graph_data: PyG Data object with node predictions
            output_shape: Shape of output grid (optional, will use predicted shape if available)
            
        Returns:
            Grid as numpy array
        """
        # Check if we should use predicted shape
        if output_shape is None and hasattr(graph_data, 'shape_params'):
            # Get shape prediction parameters
            shape_params = graph_data.shape_params
            
            # If it's a batch, use the first example's parameters
            if shape_params.dim() > 1:
                shape_params = shape_params[0]
                
            # Extract parameters
            height_ratio, width_ratio, height_offset, width_offset = shape_params.cpu().detach().numpy()
            
            # Calculate predicted dimensions from original shape (assumed to be 30x30 if not specified)
            original_shape = (30, 30)  # Default assumption
            if hasattr(graph_data, 'original_shape'):
                original_shape = graph_data.original_shape
                
            predicted_height = max(1, int(round(original_shape[0] * height_ratio + height_offset)))
            predicted_width = max(1, int(round(original_shape[1] * width_ratio + width_offset)))
            
            # Use predicted shape
            output_shape = (predicted_height, predicted_width)
        
        # Default to original shape if no shape provided or predicted
        if output_shape is None:
            output_shape = (30, 30)  # Default grid size
        
        # Get predictions from graph
        if hasattr(graph_data, 'x'):
            preds = graph_data.x
            
            # Convert to class labels
            if preds.dim() == 2 and preds.size(1) > 1:
                preds = preds.argmax(dim=1)
                
            # Reshape to grid
            rows, cols = output_shape
            preds_np = preds.cpu().detach().numpy()
            
            # Create output grid
            output_grid = np.zeros(output_shape, dtype=np.int64)
            
            # Determine scaling factors to map from node indices to grid positions
            # This handles cases where the output grid is larger or smaller than the input
            scale_r = min(30, rows) / 30.0  # Scale from standard 30x30 to actual rows
            scale_c = min(30, cols) / 30.0  # Scale from standard 30x30 to actual cols
            
            # Fill the grid with predicted values
            for node_idx in range(min(len(preds_np), 900)):  # Limit to max 900 nodes (30x30)
                # Convert node index to 2D coordinates in standard 30x30 grid
                r_orig = node_idx // 30
                c_orig = node_idx % 30
                
                # Scale to output dimensions
                r_scaled = int(r_orig * scale_r)
                c_scaled = int(c_orig * scale_c)
                
                # Ensure within bounds
                if r_scaled < rows and c_scaled < cols:
                    output_grid[r_scaled, c_scaled] = preds_np[node_idx]
            
            return output_grid
        else:
            # Fallback: return zeros grid
            return np.zeros(output_shape, dtype=np.int64)
            
    def log_reasoning_step(self, module_name, prediction=None, confidence=None, time_taken=None, details=None):
        """
        Log a reasoning step to the blackboard with timestamp
        
        Args:
            module_name: Name of the reasoning module used
            prediction: Optional prediction output
            confidence: Optional confidence score
            time_taken: Optional execution time
            details: Optional dictionary with additional details
            
        Returns:
            self for method chaining
        """
        # Create reasoning step record
        step = {
            "module_name": module_name,
            "timestamp": time.time(),
            "confidence": confidence
        }
        
        # Add optional fields if provided
        if prediction is not None:
            step["prediction"] = prediction
        if time_taken is not None:
            step["time_taken"] = time_taken
        if details is not None:
            step["details"] = details
        
        # Add to reasoning history
        if not hasattr(self.blackboard, "reasoning_history"):
            self.blackboard.reasoning_history = []
        
        self.blackboard.reasoning_history.append(step)
        
        # Track confidence scores
        if confidence is not None:
            if not hasattr(self.blackboard, "confidence_scores"):
                self.blackboard.confidence_scores = {}
            
            if module_name not in self.blackboard.confidence_scores:
                self.blackboard.confidence_scores[module_name] = []
            
            self.blackboard.confidence_scores[module_name].append(confidence)
        
        return self

    def get_reasoning_history(self, module_name=None, step_type=None, min_confidence=None):
        """
        Get reasoning history from blackboard with optional filtering
        
        Args:
            module_name: Optional module name to filter by
            step_type: Optional step type to filter by (from details["type"])
            min_confidence: Optional minimum confidence threshold
            
        Returns:
            List of reasoning steps matching the criteria
        """
        # Ensure reasoning history exists
        if not hasattr(self.blackboard, "reasoning_history"):
            return []
        
        history = self.blackboard.reasoning_history
        filtered_history = []
        
        for step in history:
            # Apply module filter
            if module_name and step.get("module_name") != module_name:
                continue
                
            # Apply confidence filter
            if min_confidence is not None and (step.get("confidence") is None or step.get("confidence") < min_confidence):
                continue
                
            # Apply step type filter
            if step_type and (not step.get("details") or step.get("details", {}).get("type") != step_type):
                continue
                
            filtered_history.append(step)
        
        return filtered_history


    def get_module_confidence(self, module_name=None):
        """
        Get average confidence score by module
        
        Args:
            module_name: Optional specific module to get confidence for
            
        Returns:
            Dictionary mapping module names to average confidence scores,
            or a single score if module_name is provided
        """
        if not hasattr(self.blackboard, "confidence_scores"):
            return {} if module_name is None else None
        
        confidence_scores = self.blackboard.confidence_scores
        
        if module_name is not None:
            scores = confidence_scores.get(module_name, [])
            return sum(scores) / len(scores) if scores else None
        
        # Calculate average for all modules
        result = {}
        for module, scores in confidence_scores.items():
            if scores:
                result[module] = sum(scores) / len(scores)
        
        return result