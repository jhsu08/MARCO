import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque
import os
import time
import json
import copy
import traceback

# Define experience replay memory
Experience = namedtuple('Experience', ('embedding', 'action', 'reward', 'next_embedding', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Experience(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Embedding network for creating task embeddings
class TaskEmbeddingNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim=128, hidden_dim=256):
        super(TaskEmbeddingNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Feature embedding network
        self.embedding_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        embeddings = self.embedding_network(x)
        return nn.functional.normalize(embeddings, p=2, dim=1)

class ProtoModuleNetwork(nn.Module):
    def __init__(self, embedding_dim, action_size, num_prototypes=10, temperature=0.5):
        super(ProtoModuleNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.action_size = action_size
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        
        # Initialize prototype vectors
        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, embedding_dim), requires_grad=True
        )
        
        # Normalize prototypes
        with torch.no_grad():
            self.prototypes.data = nn.functional.normalize(self.prototypes.data, p=2, dim=1)
        
        # Action mapping for each prototype
        self.prototype_actions = nn.Parameter(
            torch.randn(num_prototypes, action_size), requires_grad=True
        )
    
    def forward(self, embeddings):
        """
        Forward pass with robust dimension handling
        
        Args:
            embeddings: Task embeddings [batch_size, embedding_dim]
            
        Returns:
            tuple of (q_values, prototype_weights, similarities)
        """
        # Ensure embeddings has proper dimensions
        if embeddings.dim() == 0:  # Scalar tensor
            embeddings = embeddings.unsqueeze(0).unsqueeze(0)  # Add batch and feature dims
        elif embeddings.dim() == 1:  # Vector tensor without batch dimension
            embeddings = embeddings.unsqueeze(0)  # Add batch dimension
        
        # Check if embedding dimension matches prototype dimension
        if embeddings.size(-1) != self.embedding_dim:
            # Handle dimension mismatch by projecting to correct dimension
            print(f"Warning: Embedding dimension mismatch. Got {embeddings.size(-1)}, expected {self.embedding_dim}")
            
            if embeddings.size(-1) > self.embedding_dim:
                # If embedding is larger, truncate to match prototype dimension
                embeddings = embeddings[:, :self.embedding_dim]
            else:
                # If embedding is smaller, pad with zeros
                padding = torch.zeros(embeddings.size(0), self.embedding_dim - embeddings.size(-1), 
                                     device=embeddings.device)
                embeddings = torch.cat([embeddings, padding], dim=1)
        
        # Safety check for extremely large embedding dimensions that might cause memory issues
        if embeddings.size(-1) > 10000:
            # Apply dimensionality reduction via random projection if too large
            projection_matrix = torch.randn(embeddings.size(-1), self.embedding_dim, 
                                           device=embeddings.device)
            projection_matrix = nn.functional.normalize(projection_matrix, p=2, dim=0)
            embeddings = torch.matmul(embeddings, projection_matrix)
        
        # Compute distance to each prototype
        # Shape: (batch_size, num_prototypes)
        try:
            similarities = torch.matmul(embeddings, self.prototypes.t())
        except RuntimeError as e:
            print(f"Matrix multiplication error: {e}")
            print(f"Embeddings shape: {embeddings.shape}, Prototypes shape: {self.prototypes.shape}")
            
            # Emergency fallback - reshape embeddings to match prototype dimension
            reshaped_emb = embeddings.view(embeddings.size(0), -1)
            if reshaped_emb.size(1) != self.embedding_dim:
                # Apply random projection as fallback
                projection_matrix = torch.randn(reshaped_emb.size(1), self.embedding_dim, 
                                               device=reshaped_emb.device)
                projection_matrix = nn.functional.normalize(projection_matrix, p=2, dim=0)
                reshaped_emb = torch.matmul(reshaped_emb, projection_matrix)
            
            similarities = torch.matmul(reshaped_emb, self.prototypes.t())
        
        # Apply temperature scaling to similarities
        scaled_similarities = similarities / self.temperature
        
        # Convert similarities to weights using softmax
        # For 1D input we need to handle the dimension for softmax carefully
        if embeddings.size(0) == 1:
            # Only one sample, use dim=1
            prototype_weights = nn.functional.softmax(scaled_similarities, dim=1)
        else:
            # Multiple samples, use dim=1 (across prototypes for each sample)
            prototype_weights = nn.functional.softmax(scaled_similarities, dim=1)
        
        # Compute weighted sum of prototype actions
        # Shape: (batch_size, num_prototypes) × (num_prototypes, action_size) = (batch_size, action_size)
        q_values = torch.matmul(prototype_weights, self.prototype_actions)
        
        return q_values, prototype_weights, similarities

class SequenceDecisionNetwork(nn.Module):
    def __init__(self, embedding_dim, action_size, hidden_dim=128):
        super(SequenceDecisionNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.action_size = action_size
        
        # Network that takes embedding and previous actions to decide whether to continue
        self.decision_network = nn.Sequential(
            nn.Linear(embedding_dim + action_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)  # Binary decision: continue or stop
        )

    def forward(self, embeddings, prev_actions):
        """
        Forward pass with dimension and batch size checks
        
        Args:
            embeddings: Task embeddings
            prev_actions: Previous actions tensor
            
        Returns:
            Decision logits
        """
        # Ensure both tensors have at least 2 dimensions
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)  # Add batch dimension
            
        if prev_actions.dim() == 1:
            prev_actions = prev_actions.unsqueeze(0)  # Add batch dimension
        
        # Ensure batch sizes match exactly - critical fix
        batch_size_embeddings = embeddings.size(0)
        batch_size_actions = prev_actions.size(0)
        
        if batch_size_embeddings != batch_size_actions:
            # Handle batch size mismatch properly
            if batch_size_embeddings == 1:
                # Expand embeddings to match actions batch size
                embeddings = embeddings.expand(batch_size_actions, -1)
            elif batch_size_actions == 1:
                # Expand actions to match embeddings batch size
                prev_actions = prev_actions.expand(batch_size_embeddings, -1)
            else:
                # If neither is size 1, we need to truncate to smaller batch
                min_batch = min(batch_size_embeddings, batch_size_actions)
                embeddings = embeddings[:min_batch]
                prev_actions = prev_actions[:min_batch]
        
        # Final dimension check - ensure we don't have any dimension mismatch
        if embeddings.size(0) != prev_actions.size(0):
            raise ValueError(f"Batch dimensions still don't match after adjustment: embeddings {embeddings.size()} vs actions {prev_actions.size()}")
        
        # Combine embeddings with previous action information
        combined = torch.cat([embeddings, prev_actions], dim=1)
        
        # Predict continuation decision
        return self.decision_network(combined)

class ProtoMetaReasoningSystem:
    """
    Enhanced meta-reasoning system that uses Prototypical Networks to select sequences of
    reasoning modules based on task characteristics and blackboard state.
    """
    
    def __init__(self, reasoning_modules=None, feature_extractor=None, config=None):
        # Store reasoning modules
        self.reasoning_modules = reasoning_modules or {}
        self.module_names = list(self.reasoning_modules.keys())
        
        # Feature extractor to get state representation from tasks
        self.feature_extractor = feature_extractor or self.extract_task_features
        
        # Default configuration
        self.config = {
            "replay_memory_size": 10000,
            "batch_size": 64,
            "gamma": 0.99,              # Discount factor
            "learning_rate": 0.001,
            "update_target_every": 10,  # Episodes between target network updates
            "epsilon_start": 1.0,       # Starting exploration rate
            "epsilon_end": 0.01,        # Minimum exploration rate
            "epsilon_decay": 0.995,     # Decay factor for exploration rate
            "state_size": 30,           # Size of state vector (task features)
            "embedding_dim": 128,       # Size of task embeddings
            "num_prototypes": 20,       # Number of prototypes to learn
            "temperature": 0.5,         # Temperature for prototype weighting
            "log_path": "logs/",
            "model_path": "models/",
            "hidden_dim": 256,          # Size of hidden layers in networks
            "use_double_dqn": True,     # Whether to use Double DQN
            "max_sequence_length": 3,   # Maximum number of modules to apply
            "min_confidence_threshold": 0.1,  # Threshold for applying additional modules
            "blackboard_feature_size": 10,  # Size of features extracted from blackboard
            "prototype_update_rate": 0.01,  # Rate for updating prototypes
        }
        
        # Update with custom config
        if config:
            self.config.update(config)
            
        # Create directories
        os.makedirs(self.config["log_path"], exist_ok=True)
        os.makedirs(self.config["model_path"], exist_ok=True)
        
        # Initialize network components
        self.state_size = self.config["state_size"] + self.config["blackboard_feature_size"]
        self.action_size = len(self.reasoning_modules)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Task embedding network
        self.embedding_network = TaskEmbeddingNetwork(
            input_dim=self.state_size,
            embedding_dim=self.config["embedding_dim"],
            hidden_dim=self.config["hidden_dim"]
        ).to(self.device)
        
        # Prototype network for module selection (policy network)
        self.proto_policy_net = ProtoModuleNetwork(
            embedding_dim=self.config["embedding_dim"],
            action_size=self.action_size,
            num_prototypes=self.config["num_prototypes"],
            temperature=self.config["temperature"]
        ).to(self.device)
        
        # Target prototype network for stable learning
        self.proto_target_net = ProtoModuleNetwork(
            embedding_dim=self.config["embedding_dim"],
            action_size=self.action_size,
            num_prototypes=self.config["num_prototypes"],
            temperature=self.config["temperature"]
        ).to(self.device)
        
        self.proto_target_net.load_state_dict(self.proto_policy_net.state_dict())
        self.proto_target_net.eval()
        
        # Sequence decision network
        self.sequence_net = SequenceDecisionNetwork(
            embedding_dim=self.config["embedding_dim"],
            action_size=self.action_size,
            hidden_dim=self.config["hidden_dim"]
        ).to(self.device)
        
        # Optimizers
        self.embedding_optimizer = optim.Adam(
            self.embedding_network.parameters(), 
            lr=self.config["learning_rate"]
        )
        
        self.proto_optimizer = optim.Adam(
            self.proto_policy_net.parameters(), 
            lr=self.config["learning_rate"]
        )
        
        self.sequence_optimizer = optim.Adam(
            self.sequence_net.parameters(), 
            lr=self.config["learning_rate"]
        )
        
        # Loss functions
        self.criterion = nn.MSELoss()
        self.sequence_criterion = nn.CrossEntropyLoss()
        
        # Replay memory
        self.memory = ReplayMemory(self.config["replay_memory_size"])
        
        # Training parameters
        self.epsilon = self.config["epsilon_start"]
        self.episode_count = 0
        
        # Performance tracking
        self.module_performance = {name: {"accuracy": [], "time": []} 
                                  for name in self.reasoning_modules}
        self.sequence_performance = {}  # Track performance of module sequences
        
        # Prototype memory - maps prototypes to successful module sequences
        self.prototype_memory = []
        
        # Task history
        self.task_history = []
    
    def extract_task_features(self, task, feature_size=40):
        """Enhanced feature extraction from a Task object for embedding training"""
        features = []
        
        # 1. Color/value distribution features
        try:
            color_stats = []
            num_colors = []
            
            for input_grid, _ in task.train_pairs:
                grid_array = np.array(input_grid)
                values, counts = np.unique(grid_array, return_counts=True)
                
                # Calculate color entropy
                total = sum(counts)
                if total > 0:
                    distribution = [c/total for c in counts]
                    entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in distribution)
                    color_stats.append(entropy / 3.32)  # Normalize by log2(10)
                
                # Number of colors
                num_colors.append(len(values) / 10.0)  # Normalize by max expected colors
            
            # Add color features
            features.append(np.mean(color_stats) if color_stats else 0.0)
            features.append(np.mean(num_colors) if num_colors else 0.0)
        except Exception as e:
            print("Using default value distribution features")
            features.extend([0.0, 0.0])
        
        # 2. Grid size features
        if task.train_pairs:
            input_shape = np.array(task.train_pairs[0][0]).shape
            features.append(input_shape[0] / 30.0)  # Normalized row count
            features.append(input_shape[1] / 30.0)  # Normalized column count
            features.append((input_shape[0] * input_shape[1]) / 900.0)  # Normalized total size
        else:
            print("Using default grid size features")
            features.extend([0.0, 0.0, 0.0])
        
        # 3. Basic change analysis (without the duplicate edge transformation features)
        if task.train_pairs:
            # Basic change analysis
            avg_changes = 0
            for input_grid, output_grid in task.train_pairs:
                input_np = np.array(input_grid)
                output_np = np.array(output_grid)
                
                # Find shape intersection
                min_rows = min(input_np.shape[0], output_np.shape[0])
                min_cols = min(input_np.shape[1], output_np.shape[1])
                
                # Count changes
                if min_rows > 0 and min_cols > 0:
                    changes = np.sum(input_np[:min_rows, :min_cols] != output_np[:min_rows, :min_cols])
                    total_cells = min_rows * min_cols
                    change_ratio = changes / total_cells if total_cells > 0 else 0
                    avg_changes += change_ratio
            
            if task.train_pairs:
                avg_changes /= len(task.train_pairs)
            features.append(avg_changes)
        else:
            print("Using default change features")
            features.append(0.0)  # Default change ratio
        
        # 4. Pattern complexity metrics
        if task.train_pairs:
            # Calculate average symmetry scores
            symmetry_scores = []
            for input_grid, _ in task.train_pairs:
                grid_array = np.array(input_grid)
                
                # Horizontal symmetry
                rows, cols = grid_array.shape
                h_symmetry = 0
                v_symmetry = 0
                
                if cols >= 2:
                    # Check horizontal symmetry (left-right)
                    mid = cols // 2
                    left_side = grid_array[:, :mid]
                    right_side = grid_array[:, cols-1:mid-1 if mid > 0 else None:-1]
                    if left_side.shape == right_side.shape:
                        h_symmetry = np.mean(left_side == right_side)
                
                if rows >= 2:
                    # Check vertical symmetry (top-bottom)
                    mid = rows // 2
                    top_side = grid_array[:mid, :]
                    bottom_side = grid_array[rows-1:mid-1 if mid > 0 else None:-1, :]
                    if top_side.shape == bottom_side.shape:
                        v_symmetry = np.mean(top_side == bottom_side)
                
                symmetry_scores.append((h_symmetry + v_symmetry) / 2)
            
            # Add average symmetry score
            features.append(np.mean(symmetry_scores) if symmetry_scores else 0.0)
            
            # Add count of different shapes/patterns (connected components)
            avg_components = 0
            for input_grid, _ in task.train_pairs:
                grid_array = np.array(input_grid)
                # Simple connected components count (non-zero elements)
                from scipy.ndimage import label
                non_zero_mask = (grid_array > 0)
                if np.any(non_zero_mask):
                    labeled_array, num_components = label(non_zero_mask)
                    avg_components += num_components / 20.0  # Normalize by expected max
                
            features.append(avg_components / len(task.train_pairs) if task.train_pairs else 0.0)
        else:
            print("Using default pattern complexity features")
            features.extend([0.0, 0.0])  # Default pattern metrics
        
        # 5. Task metadata
        features.append(len(task.train_pairs) / 10.0)  # Number of training examples
        features.append(len(task.test_pairs) / 10.0)   # Number of test examples
    
        # 6. Graph information analysis (edge types and transformations)
        if hasattr(task, 'train_graphs') and task.train_graphs:
            # Track edge information
            edge_types_counts = {}
            edge_transformation_counts = {"unchanged": 0, "removed": 0, "added": 0}
            total_nodes = 0
            total_edges = 0
            total_transformations = 0
            
            for graph in task.train_graphs:
                # Count nodes
                if hasattr(graph, 'x'):
                    total_nodes += graph.x.size(0)
                
                # Count total edges (only using edge_index for total)
                if hasattr(graph, 'edge_index'):
                    total_edges += getattr(graph, 'edge_index').size(1) // 2  # Divide by 2 because edges are bidirectional
                
                # Count edges by specific type (without double-counting)
                specific_edge_types = ['value_edge_index', 'region_edge_index', 'contextual_edge_index', 'alignment_edge_index']
                for edge_type in specific_edge_types:
                    if hasattr(graph, edge_type):
                        edge_count = getattr(graph, edge_type).size(1) // 2
                        if edge_type not in edge_types_counts:
                            edge_types_counts[edge_type] = 0
                        edge_types_counts[edge_type] += edge_count
                
                # Add transformation information if available
                if hasattr(graph, 'edge_transformation_labels'):
                    labels = graph.edge_transformation_labels.cpu().numpy() if torch.is_tensor(graph.edge_transformation_labels) else graph.edge_transformation_labels
                    
                    if len(labels) > 0:
                        # Count transformation types (assuming 0=unchanged, 1=removed, 2=added)
                        edge_transformation_counts["unchanged"] += np.sum(labels == 0)
                        edge_transformation_counts["removed"] += np.sum(labels == 1) 
                        edge_transformation_counts["added"] += np.sum(labels == 2)
                        total_transformations += len(labels)
            
            # Add node density feature
            num_graphs = len(task.train_graphs)
            features.append(total_nodes / (900 * num_graphs) if num_graphs > 0 else 0.0)  # Average normalized node count
            
            # Add total edge density feature
            features.append(total_edges / (4000 * num_graphs) if num_graphs > 0 else 0.0)  # Normalized edge count
            
            # Add specific edge type proportion features
            specific_edge_types = ['value_edge_index', 'region_edge_index', 'contextual_edge_index', 'alignment_edge_index']
            for edge_type in specific_edge_types:
                if total_edges > 0:
                    ratio = edge_types_counts.get(edge_type, 0) / total_edges
                else:
                    ratio = 0.0
                features.append(ratio)
            
            # Add transformation distribution features
            if total_transformations > 0:
                for key in ["unchanged", "removed", "added"]:
                    features.append(edge_transformation_counts[key] / total_transformations)
            else:
                features.extend([0.0, 0.0, 0.0])  # Default values if no transformations
        else:
            print("Using default graph features")
            features.append(0.0)  # Default node density
            features.append(0.0)  # Default edge density
            features.extend([0.0, 0.0, 0.0, 0.0])  # Default edge type features
            features.extend([0.0, 0.0, 0.0])  # Default transformation features
        
        # Ensure we have the right number of features
        while len(features) < feature_size:
            features.append(0.0)
        
        # Truncate if too long
        features = features[:feature_size]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def extract_blackboard_features(self, task):
        """
        Extract features from the blackboard state
        
        Args:
            task: Task object with blackboard
            
        Returns:
            List of blackboard features
        """
        features = []
        
        # Initialize with defaults
        num_predicates = 0
        num_transformations = 0
        num_text_entries = 0
        reasoning_steps = 0
        confidence_avg = 0.5
        has_nlm_transforms = 0
        has_neurosymbolic_transforms = 0
        has_llm_transforms = 0
        recent_module_name = "none"
        blackboard_size = 0
        module_diversity = 0
        repeated_modules = 0
        
        # Only extract if blackboard exists
        if hasattr(task, 'blackboard'):
            bb = task.blackboard
            
            # Count predicates (adapt to your blackboard structure)
            if hasattr(bb, 'logical_predicates'):
                num_predicates = len(bb.logical_predicates) / 100.0  # Normalize
            elif hasattr(bb, 'knowledge_base'):
                predicate_count = sum(1 for k in bb.knowledge_base.keys() if 'predicate' in k)
                num_predicates = predicate_count / 100.0
            
            # Count transformations (adapt to your blackboard structure)
            if hasattr(bb, 'transformations'):
                # If transformations is a dict of source -> transforms
                if isinstance(bb.transformations, dict):
                    for source, transforms in bb.transformations.items():
                        num_transformations += len(transforms)
                        
                        # Check for specific module transformations
                        if source == 'nlm' and transforms:
                            has_nlm_transforms = 1.0
                        elif source == 'neurosymbolic' and transforms:
                            has_neurosymbolic_transforms = 1.0
                        elif source == 'llm' and transforms:
                            has_llm_transforms = 1.0
                else:
                    # If transformations is a list/object
                    num_transformations = len(bb.transformations) / 50.0
            elif hasattr(bb, 'knowledge_base'):
                transform_count = sum(1 for k in bb.knowledge_base.keys() if 'transform' in k)
                num_transformations = transform_count / 50.0
                
                # Check for specific source transformations
                for k, v in bb.knowledge_sources.items():
                    if 'transform' in k:
                        if v == 'nlm':
                            has_nlm_transforms = 1.0
                        elif v == 'neurosymbolic' or v == 'unified':
                            has_neurosymbolic_transforms = 1.0
                        elif v == 'llm':
                            has_llm_transforms = 1.0
            
            # Count text entries
            if hasattr(bb, 'textual_data'):
                num_text_entries = len(bb.textual_data) / 10.0  # Normalize
            
            # Reasoning history
            if hasattr(bb, 'reasoning_history'):
                reasoning_steps = len(bb.reasoning_history) / 10.0  # Normalize
                
                # Most recent module
                if bb.reasoning_history:
                    recent_module = bb.reasoning_history[-1].get("module_name")
                    if recent_module in self.module_names:
                        recent_module_name = recent_module
                
                # Module diversity and repetition
                if bb.reasoning_history:
                    # Count distinct modules used
                    used_modules = set()
                    module_counts = {}
                    
                    for step in bb.reasoning_history:
                        module = step.get("module_name")
                        if module:
                            used_modules.add(module)
                            module_counts[module] = module_counts.get(module, 0) + 1
                    
                    # Calculate diversity (number of distinct modules / total modules used)
                    module_diversity = len(used_modules) / max(len(self.module_names), 1)
                    
                    # Calculate repetition (max count of any module / total steps)
                    max_count = max(module_counts.values()) if module_counts else 0
                    repeated_modules = max_count / max(len(bb.reasoning_history), 1)
            
            # Average confidence
            if hasattr(bb, 'confidence_scores'):
                if isinstance(bb.confidence_scores, dict):
                    confidence_values = []
                    for scores in bb.confidence_scores.values():
                        if isinstance(scores, list):
                            confidence_values.extend(scores)
                        else:
                            confidence_values.append(scores)
                    
                    confidence_avg = sum(confidence_values) / len(confidence_values) if confidence_values else 0.5
                else:
                    confidence_avg = bb.confidence_scores
            
            # Estimate total size of blackboard data
            try:
                if hasattr(bb, 'knowledge_base'):
                    blackboard_size = len(str(bb.knowledge_base)) / 10000.0  # Normalize
                else:
                    # Fallback estimation
                    blackboard_size = (
                        num_predicates * 100 + 
                        num_transformations * 50 + 
                        num_text_entries * 100
                    ) / 10000.0
            except:
                blackboard_size = 0.0
        
        # Add features
        features.extend([
            num_predicates,
            num_transformations,
            num_text_entries,
            reasoning_steps,
            confidence_avg,
            has_nlm_transforms,
            has_neurosymbolic_transforms,
            has_llm_transforms,
            self.module_names.index(recent_module_name) / max(1, len(self.module_names)) if recent_module_name in self.module_names else 0.0,
            blackboard_size,
            module_diversity,
            repeated_modules
        ])
        
        # Ensure we have the right number of features
        while len(features) < self.config["blackboard_feature_size"]:
            features.append(0.0)
        
        # Truncate if too long
        features = features[:self.config["blackboard_feature_size"]]
        
        return features
    
    def add_reasoning_module(self, name, module):
        """Add a reasoning module to the system"""
        self.reasoning_modules[name] = module
        self.module_names.append(name)
        self.module_performance[name] = {"accuracy": [], "time": []}
        
        # Update action space
        self.action_size = len(self.reasoning_modules)
        
        # Reinitialize networks with new action size
        old_proto_policy_net = self.proto_policy_net
        old_proto_target_net = self.proto_target_net
        old_sequence_net = self.sequence_net
        
        # Create new prototype networks
        self.proto_policy_net = ProtoModuleNetwork(
            embedding_dim=self.config["embedding_dim"],
            action_size=self.action_size,
            num_prototypes=self.config["num_prototypes"],
            temperature=self.config["temperature"]
        ).to(self.device)
        
        self.proto_target_net = ProtoModuleNetwork(
            embedding_dim=self.config["embedding_dim"],
            action_size=self.action_size,
            num_prototypes=self.config["num_prototypes"],
            temperature=self.config["temperature"]
        ).to(self.device)
        
        # Create new sequence network
        self.sequence_net = SequenceDecisionNetwork(
            embedding_dim=self.config["embedding_dim"],
            action_size=self.action_size,
            hidden_dim=self.config["hidden_dim"]
        ).to(self.device)
        
        # Try to copy weights for existing modules
        # Copy prototype parameters where possible
        with torch.no_grad():
            # Copy shared prototype embeddings
            old_prototypes = old_proto_policy_net.prototypes.data
            self.proto_policy_net.prototypes.data[:old_prototypes.size(0)] = old_prototypes
            
            # Copy and pad prototype actions
            old_actions = old_proto_policy_net.prototype_actions.data
            old_action_size = old_actions.size(1)
            self.proto_policy_net.prototype_actions.data[:, :old_action_size] = old_actions
        
        # Update target network
        self.proto_target_net.load_state_dict(self.proto_policy_net.state_dict())
        
        # Update optimizers
        self.proto_optimizer = optim.Adam(
            self.proto_policy_net.parameters(), 
            lr=self.config["learning_rate"]
        )
        
        self.sequence_optimizer = optim.Adam(
            self.sequence_net.parameters(), 
            lr=self.config["learning_rate"]
        )
    
    def get_task_embedding(self, task):
        """
        Get embedding vector for a task
        
        Args:
            task: Task object
            
        Returns:
            Task embedding tensor
        """
        # Extract features
        features = self.feature_extractor(task)
        
        # Get embedding
        with torch.no_grad():
            embedding = self.embedding_network(features)
        
        return embedding
    
    def select_module_sequence(self, task):
        """
        Select a sequence of reasoning modules using prototypical networks
        
        Args:
            task: Task object to solve
            
        Returns:
            List of selected module names
        """
        try:
            # Extract task features and compute embedding
            features = self.feature_extractor(task)
            embedding = self.embedding_network(features)
            
            # Ensure embedding has proper dimensions for prototype network
            if embedding.dim() == 0:  # Scalar tensor
                embedding = embedding.unsqueeze(0).unsqueeze(0)  # Add batch and feature dims
            elif embedding.dim() == 1:  # Vector tensor without batch dimension
                embedding = embedding.unsqueeze(0)  # Add batch dimension
            
            # Initialize sequence
            module_sequence = []
            
            # Keep track of previous actions for sequence decision
            prev_actions_tensor = torch.zeros(1, self.action_size, device=self.device)
            
            # Apply modules until max sequence length
            for step in range(self.config["max_sequence_length"]):
                # Select module with epsilon-greedy
                if random.random() < self.epsilon:
                    # Explore: randomly select a module
                    action = random.randrange(self.action_size)
                else:
                    # Exploit: select best module based on prototypes
                    with torch.no_grad():
                        if step == 0:
                            # First module selection (no sequence info yet)
                            q_values, proto_weights, _ = self.proto_policy_net(embedding)
                            action = q_values.max(1)[1].item()
                        else:
                            # For subsequent modules, decide whether to continue
                            continue_logits = self.sequence_net(embedding, prev_actions_tensor)
                            continue_prob = torch.softmax(continue_logits, dim=1)[0, 1].item()
                            
                            if continue_prob < 0.5:
                                # Stop sequence if model suggests stopping
                                break
                            
                            # Get next action (avoid repeating modules)
                            q_values, proto_weights, _ = self.proto_policy_net(embedding)
                            q_copy = q_values.clone()
                            
                            # Mask previously used modules
                            for i in range(self.action_size):
                                if prev_actions_tensor[0, i] > 0:
                                    q_copy[0, i] = -float('inf')
                                    
                            action = q_copy.max(1)[1].item()
                
                # Add selected module to sequence
                module_name = self.module_names[action]
                module_sequence.append(module_name)
                
                # Update previous actions tensor
                prev_actions_tensor[0, action] = 1.0
            
            # Decay epsilon
            self.epsilon = max(
                self.config["epsilon_end"],
                self.epsilon * self.config["epsilon_decay"]
            )
            
            # Store selected sequence in prototype memory with task embedding
            if len(module_sequence) > 0:
                self.prototype_memory.append({
                    "embedding": embedding.detach(),
                    "sequence": module_sequence,
                    "performance": None  # Will be updated after execution
                })
            
            return module_sequence
            
        except Exception as e:
            # Fallback to a simple strategy if there's an error
            print(f"Error in select_module_sequence: {e}")
            # Return a single random module as fallback
            if self.module_names:
                return [random.choice(self.module_names)]
            return []
        
    def optimize_embeddings(self, embedding, action, reward, next_embedding, done):
        """
        Train the embedding network to create more useful task representations
        
        Args:
            embedding: Current task embedding
            action: Selected action
            reward: Reward received
            next_embedding: Next task embedding
            done: Whether episode is done
        """
        # Sample batch
        if len(self.memory) < self.config["batch_size"]:
            return
            
        experiences = self.memory.sample(self.config["batch_size"])
        batch = Experience(*zip(*experiences))
        
        # Prepare batch tensors
        embedding_batch = torch.cat(batch.embedding)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_embedding_batch = torch.cat(batch.next_embedding)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # Compute Q values for current embeddings
        q_values, _, _ = self.proto_policy_net(embedding_batch)
        current_q = q_values.gather(1, action_batch)
        
        # Compute expected Q values
        with torch.no_grad():
            if self.config["use_double_dqn"]:
                # Double DQN: use policy net for action selection, target net for Q values
                next_actions = self.proto_policy_net(next_embedding_batch)[0].max(1)[1].unsqueeze(1)
                next_q = self.proto_target_net(next_embedding_batch)[0].gather(1, next_actions)
            else:
                # Standard DQN
                next_q = self.proto_target_net(next_embedding_batch)[0].max(1)[0].unsqueeze(1)
                
            expected_q = reward_batch + (1 - done_batch) * self.config["gamma"] * next_q
        
        # Compute prototype network loss
        proto_loss = self.criterion(current_q, expected_q)
        
        # Update networks
        self.embedding_optimizer.zero_grad()
        self.proto_optimizer.zero_grad()
        
        proto_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.embedding_network.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.proto_policy_net.parameters(), 1.0)
        
        self.embedding_optimizer.step()
        self.proto_optimizer.step()
        
        return proto_loss.item()
    
    def optimize_sequence_network(self, embedding, prev_actions, continue_label):
        """
        Train the sequence continuation network
        
        Args:
            embedding: Task embedding
            prev_actions: Previous actions tensor
            continue_label: Binary label (1=continue, 0=stop)
        """
        # Need at least one sample
        if not hasattr(self, 'sequence_memory'):
            self.sequence_memory = []
            
        # Clone the tensors to avoid backward graph issues
        embedding_copy = embedding.detach().clone()
        prev_actions_copy = prev_actions.detach().clone()
        
        # Add to memory
        self.sequence_memory.append((embedding_copy, prev_actions_copy, continue_label))
        
        # Need enough samples to train
        if len(self.sequence_memory) < 32:
            return 0.0
            
        # Sample batch
        batch = random.sample(self.sequence_memory, min(32, len(self.sequence_memory)))
        
        # Check if batch is empty (should never happen but just in case)
        if not batch:
            return 0.0
        
        # Prepare batch tensors - ensure all have same batch dimension
        embeddings = [item[0] for item in batch]
        prev_actions_list = [item[1] for item in batch]
        continue_labels = torch.tensor([item[2] for item in batch], dtype=torch.long, device=self.device)
        
        # Check that we have valid tensors
        if not embeddings or not prev_actions_list:
            return 0.0
        
        # Ensure all embeddings have the same shape
        # Handle case where embedding might be a scalar or 1D tensor
        for i, emb in enumerate(embeddings):
            if emb.dim() == 0:  # Handle scalar tensor
                embeddings[i] = emb.unsqueeze(0).unsqueeze(0)  # Add batch and feature dimension
            elif emb.dim() == 1:  # Handle 1D tensor
                embeddings[i] = emb.unsqueeze(0)  # Add batch dimension
        
        embedding_shape = embeddings[0].shape[1] if embeddings[0].dim() > 1 else 1  # Get feature dimension safely
        
        for i, emb in enumerate(embeddings):
            if emb.dim() == 1:
                embeddings[i] = emb.unsqueeze(0)  # Add batch dimension if missing
            
            # If we still don't have 2D tensor, reshape it
            if embeddings[i].dim() < 2:
                embeddings[i] = embeddings[i].reshape(1, -1)
                
            if embeddings[i].shape[1] != embedding_shape:
                # Handle possible dimension mismatch by padding or truncating
                if embeddings[i].shape[1] < embedding_shape:
                    # Pad
                    padding = torch.zeros(embeddings[i].shape[0], embedding_shape - embeddings[i].shape[1], device=self.device)
                    embeddings[i] = torch.cat([embeddings[i], padding], dim=1)
                else:
                    # Truncate
                    embeddings[i] = embeddings[i][:, :embedding_shape]
        
        # Ensure all action tensors have the same shape
        for i, act in enumerate(prev_actions_list):
            if act.dim() == 0:  # Handle scalar tensor
                prev_actions_list[i] = act.unsqueeze(0).unsqueeze(0)  # Add batch and feature dimension
            elif act.dim() == 1:  # Handle 1D tensor
                prev_actions_list[i] = act.unsqueeze(0)  # Add batch dimension
        
        action_shape = prev_actions_list[0].shape[1] if prev_actions_list[0].dim() > 1 else 1
        
        for i, act in enumerate(prev_actions_list):
            if act.dim() == 1:
                prev_actions_list[i] = act.unsqueeze(0)  # Add batch dimension if missing
                
            # If we still don't have 2D tensor, reshape it
            if prev_actions_list[i].dim() < 2:
                prev_actions_list[i] = prev_actions_list[i].reshape(1, -1)
                
            if prev_actions_list[i].shape[1] != action_shape:
                # Handle possible dimension mismatch
                if prev_actions_list[i].shape[1] < action_shape:
                    # Pad
                    padding = torch.zeros(prev_actions_list[i].shape[0], action_shape - prev_actions_list[i].shape[1], device=self.device)
                    prev_actions_list[i] = torch.cat([prev_actions_list[i], padding], dim=1)
                else:
                    # Truncate
                    prev_actions_list[i] = prev_actions_list[i][:, :action_shape]
        
        # Stack with careful dimension handling
        try:
            embedding_batch = torch.cat(embeddings, dim=0)
            prev_actions_batch = torch.cat(prev_actions_list, dim=0)
            
            # Ensure batch sizes match
            batch_size = min(embedding_batch.size(0), prev_actions_batch.size(0), continue_labels.size(0))
            embedding_batch = embedding_batch[:batch_size]
            prev_actions_batch = prev_actions_batch[:batch_size]
            continue_labels = continue_labels[:batch_size]
            
            # Forward pass
            continue_logits = self.sequence_net(embedding_batch, prev_actions_batch)
            
            # Compute loss
            sequence_loss = self.sequence_criterion(continue_logits, continue_labels)
            
            # Update network
            self.sequence_optimizer.zero_grad()
            try:
                sequence_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.sequence_net.parameters(), 1.0)
                self.sequence_optimizer.step()
            except RuntimeError as e:
                if "trying to backward through the graph a second time" in str(e).lower():
                    # This might happen if some tensors are still part of another computational graph
                    # Create a detached copy of the loss to prevent graph issues
                    detached_loss = sequence_loss.detach().clone()
                    detached_loss.requires_grad_(True)
                    detached_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.sequence_net.parameters(), 1.0)
                    self.sequence_optimizer.step()
                else:
                    # For other runtime errors, log and continue
                    print(f"Backward error in sequence network: {e}")
            
            return sequence_loss.item()
            
        except Exception as e:
            print(f"Error in optimize_sequence_network: {e}")
            return 0.0
    
    def optimize_model(self):
        """Train all components using batches from replay memory"""
        losses = {}
        
        # Optimize embeddings and prototype network
        if len(self.memory) >= self.config["batch_size"]:
            proto_loss = 0
            sequence_loss = 0
            
            try:
                # Sample batch
                experiences = self.memory.sample(self.config["batch_size"])
                batch = Experience(*zip(*experiences))
                
                # Prepare batch tensors
                embeddings_fixed = []
                for e in batch.embedding:
                    e = e.detach()
                    if e.dim() == 1:
                        e = e.unsqueeze(0)
                    embeddings_fixed.append(e)
                embedding_batch = torch.cat(embeddings_fixed, dim=0)
                action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
                reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
                next_embeddings_fixed = []
                for e in batch.next_embedding:
                    e = e.detach()
                    if e.dim() == 1:
                        e = e.unsqueeze(0)
                    next_embeddings_fixed.append(e)       
                next_embedding_batch = torch.cat(next_embeddings_fixed, dim=0)
                done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)
                assert embedding_batch.shape[1] == self.config["embedding_dim"], \
                    f"Expected embedding dimension {self.config['embedding_dim']}, got {embedding_batch.shape[1]}"
                # Perform embedding dimension sanity check and correction
                if embedding_batch.size(-1) != self.config["embedding_dim"]:
                    print(f"Warning: Embedding batch dimension mismatch in optimize_model. " 
                          f"Got {embedding_batch.size(-1)}, expected {self.config['embedding_dim']}")
                    
                    if embedding_batch.size(-1) > self.config["embedding_dim"]:
                        # If embedding is larger, truncate
                        embedding_batch = embedding_batch[:, :self.config["embedding_dim"]]
                        next_embedding_batch = next_embedding_batch[:, :self.config["embedding_dim"]]
                    else:
                        # If embedding is smaller, pad with zeros
                        padding_current = torch.zeros(embedding_batch.size(0), 
                                                     self.config["embedding_dim"] - embedding_batch.size(-1), 
                                                     device=self.device)
                        padding_next = torch.zeros(next_embedding_batch.size(0), 
                                                  self.config["embedding_dim"] - next_embedding_batch.size(-1), 
                                                  device=self.device)
                        embedding_batch = torch.cat([embedding_batch, padding_current], dim=1)
                        next_embedding_batch = torch.cat([next_embedding_batch, padding_next], dim=1)
                
                # Compute Q values for current embeddings
                q_values, proto_weights, similarities = self.proto_policy_net(embedding_batch)
                current_q = q_values.gather(1, action_batch)
                
                # Compute expected Q values
                with torch.no_grad():
                    if self.config["use_double_dqn"]:
                        # Double DQN: use policy net for action selection, target net for Q values
                        next_policy_q, _, _ = self.proto_policy_net(next_embedding_batch)
                        next_actions = next_policy_q.max(1)[1].unsqueeze(1)
                        next_target_q, _, _ = self.proto_target_net(next_embedding_batch)
                        next_q = next_target_q.gather(1, next_actions)
                    else:
                        # Standard DQN
                        next_q, _, _ = self.proto_target_net(next_embedding_batch)
                        next_q = next_q.max(1)[0].unsqueeze(1)
                        
                    expected_q = reward_batch + (1 - done_batch) * self.config["gamma"] * next_q
                
                # Compute prototype network loss
                proto_loss = self.criterion(current_q, expected_q)
                
                # Update networks
                self.embedding_optimizer.zero_grad()
                self.proto_optimizer.zero_grad()
                
                proto_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.embedding_network.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.proto_policy_net.parameters(), 1.0)
                
                self.embedding_optimizer.step()
                self.proto_optimizer.step()
                
                losses["proto_loss"] = proto_loss.item()
                
                # Optimize sequence network if we have sequence data
                if hasattr(self, 'sequence_memory') and len(self.sequence_memory) >= 16:
                    try:
                        # Sample sequence batch
                        sequence_batch = random.sample(self.sequence_memory, 16)
                        
                        # Prepare batch tensors - ensure correct dimensions
                        seq_embeddings = []
                        seq_prev_actions = []
                        seq_continue_labels = []
                        
                        for emb, act, label in sequence_batch:
                            # Handle embedding dimensions
                            if emb.dim() == 0:
                                emb = emb.unsqueeze(0).unsqueeze(0)
                            elif emb.dim() == 1:
                                emb = emb.unsqueeze(0)
                                
                            # Handle action dimensions
                            if act.dim() == 0:
                                act = act.unsqueeze(0).unsqueeze(0)
                            elif act.dim() == 1:
                                act = act.unsqueeze(0)
                            
                            # Add to lists
                            seq_embeddings.append(emb)
                            seq_prev_actions.append(act)
                            seq_continue_labels.append(label)
                        
                        # Concatenate tensors
                        seq_embedding_batch = torch.cat(seq_embeddings)
                        seq_prev_actions_batch = torch.cat(seq_prev_actions)
                        seq_continue_batch = torch.tensor(seq_continue_labels, dtype=torch.long, device=self.device)
                        
                        # Ensure embedding dimension matches
                        if seq_embedding_batch.size(-1) != self.config["embedding_dim"]:
                            if seq_embedding_batch.size(-1) > self.config["embedding_dim"]:
                                # If embedding is larger, truncate
                                seq_embedding_batch = seq_embedding_batch[:, :self.config["embedding_dim"]]
                            else:
                                # If embedding is smaller, pad with zeros
                                padding = torch.zeros(seq_embedding_batch.size(0), 
                                                    self.config["embedding_dim"] - seq_embedding_batch.size(-1), 
                                                    device=self.device)
                                seq_embedding_batch = torch.cat([seq_embedding_batch, padding], dim=1)
                        
                        # Forward pass
                        continue_logits = self.sequence_net(seq_embedding_batch, seq_prev_actions_batch)
                        
                        # Compute loss
                        sequence_loss = self.sequence_criterion(continue_logits, seq_continue_batch)
                        
                        # Update network
                        self.sequence_optimizer.zero_grad()
                        sequence_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.sequence_net.parameters(), 1.0)
                        self.sequence_optimizer.step()
                        
                        losses["sequence_loss"] = sequence_loss.item()
                    except Exception as e:
                        print(f"Error in sequence network optimization: {e}")
                        
            except Exception as e:
                print(f"Error in optimize_model: {e}")
                import traceback
                traceback.print_exc()
                
        return losses
    
    def solve_task(self, task, training=True):
        """
        Solve a task by selecting and applying a sequence of reasoning modules
        
        Args:
            task: Task object to solve
            training: Whether to update the networks (set to False for evaluation)
            
        Returns:
            List of predictions for test examples
        """
        # Extract features and compute embedding
        features = self.feature_extractor(task)
        embedding = self.embedding_network(features)
        
        # Task execution record
        execution = {
            "task_id": task.task_id,
            "start_time": time.time(),
            "module_sequence": None,
            "node_accuracy": None,
            "grid_accuracy": None,
            "episode": self.episode_count
        }
        
        try:
            # Select a module sequence
            if training:
                module_sequence = self.select_module_sequence(task)
            else:
                module_sequence = self.get_best_module_sequence(task)
                
            execution["module_sequence"] = module_sequence
            
            # Apply modules in sequence
            predictions = None
            current_node_accuracy = 0.0
            current_grid_accuracy = 0.0
            blackboard_snapshots = []  # Store blackboard state after each module
            
            # Process each module in the sequence
            for i, module_name in enumerate(module_sequence):
                module = self.reasoning_modules[module_name]
                
                # Take a snapshot of blackboard state before module execution
                if hasattr(task, 'blackboard'):
                    pre_state = self._capture_blackboard_state(task.blackboard)
                
                # Solve task with this module, passing previous predictions if available
                start_time = time.time()
                
                if predictions is not None and hasattr(module, 'refine'):
                    # Use refine method if available and we have previous predictions
                    current_predictions = module.refine(task, predictions)
                else:
                    # Otherwise use standard solve method
                    current_predictions = module.solve(task)
                
                execution_time = time.time() - start_time
                
                # Capture blackboard changes after module execution
                if hasattr(task, 'blackboard'):
                    post_state = self._capture_blackboard_state(task.blackboard)
                    blackboard_delta = self._compute_blackboard_delta(pre_state, post_state)
                    blackboard_snapshots.append({
                        "module": module_name,
                        "delta": blackboard_delta,
                        "state": post_state
                    })
                
                # Evaluate predictions
                current_node_accuracy, current_grid_accuracy = self.evaluate_predictions(current_predictions, task)
                
                # Log reasoning step in task if supported
                if hasattr(task, 'log_reasoning_step'):
                    task.log_reasoning_step(
                        module_name=module_name,
                        prediction=current_predictions,
                        confidence=current_node_accuracy,  # Using node accuracy as confidence
                        time_taken=execution_time,
                        details={
                            "step": i, 
                            "sequence": module_sequence,
                            "node_accuracy": current_node_accuracy,
                            "grid_accuracy": current_grid_accuracy
                        }
                    )
                
                # Update performance metrics for this module
                self.module_performance[module_name]["node_accuracy"] = self.module_performance[module_name].get("node_accuracy", []) + [current_node_accuracy]
                self.module_performance[module_name]["grid_accuracy"] = self.module_performance[module_name].get("grid_accuracy", []) + [current_grid_accuracy]
                self.module_performance[module_name]["time"].append(execution_time)
                
                # Update predictions
                predictions = current_predictions
                
                # Check if further improvement is needed
                if current_grid_accuracy > 0.95:
                    # Solution is good enough, no need for more modules
                    break
                
                # If training, prepare data for sequence network training
                if training and i < len(module_sequence) - 1:
                    # Create one-hot encoding of actions taken so far
                    prev_actions_tensor = torch.zeros(1, self.action_size, device=self.device)
                    for prev_module in module_sequence[:i+1]:
                        prev_idx = self.module_names.index(prev_module)
                        prev_actions_tensor[0, prev_idx] = 1.0
                    
                    # Label is 1 (continue) since we did add another module
                    continue_label = 1
                    
                    # Train sequence network
                    self.optimize_sequence_network(embedding, prev_actions_tensor, continue_label)
            
            # For the last module in sequence, if training, also train sequence network
            if training and len(module_sequence) > 0:
                # Create one-hot encoding of actions taken
                prev_actions_tensor = torch.zeros(1, self.action_size, device=self.device)
                for prev_module in module_sequence:
                    prev_idx = self.module_names.index(prev_module)
                    prev_actions_tensor[0, prev_idx] = 1.0
                
                # Label is 0 (stop) since we didn't add more modules
                continue_label = 0
                
                # Train sequence network
                self.optimize_sequence_network(embedding, prev_actions_tensor, continue_label)
            
            # Update sequence performance tracking
            sequence_key = "->".join(module_sequence)
            if sequence_key not in self.sequence_performance:
                self.sequence_performance[sequence_key] = {
                    "count": 0, 
                    "node_accuracy": [],
                    "grid_accuracy": []
                }
            
            self.sequence_performance[sequence_key]["count"] += 1
            self.sequence_performance[sequence_key]["node_accuracy"].append(current_node_accuracy)
            self.sequence_performance[sequence_key]["grid_accuracy"].append(current_grid_accuracy)
            
            # Set final accuracy metrics
            execution["node_accuracy"] = current_node_accuracy
            execution["grid_accuracy"] = current_grid_accuracy
            
            # Update prototype memory with performance
            for entry in self.prototype_memory:
                if entry["sequence"] == module_sequence and entry["performance"] is None:
                    entry["performance"] = (0.7*current_node_accuracy)+(0.3*current_grid_accuracy)
                    # {
                    #     "node_accuracy": current_node_accuracy,
                    #     "grid_accuracy": current_grid_accuracy
                    # }
            
            if training:
                # Store experience in replay memory
                # Convert module sequence to action index of first module
                first_action = self.module_names.index(module_sequence[0]) if module_sequence else 0
                next_embedding = self.embedding_network(features)  # Updated embedding
                
                # Use appropriate accuracy as reward (could use grid_accuracy or a combination)
                # Using grid accuracy as primary reward since it's a stronger signal of complete success
                reward = current_grid_accuracy * 0.7 + current_node_accuracy * 0.3  # Weight grid accuracy higher
                done = True  # Each task is a separate episode
                
                # Store in replay memory
                if embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)
                if next_embedding.dim() == 1:
                    next_embedding = next_embedding.unsqueeze(0)
                # print(f"Storing embedding shape: {embedding.shape}, next_embedding shape: {next_embedding.shape}")
                self.memory.push(embedding, first_action, reward, next_embedding, done)
                
                # Train the model
                losses = self.optimize_model()
                execution["losses"] = losses
                
                # Update target network
                self.episode_count += 1
                if self.episode_count % self.config["update_target_every"] == 0:
                    self.proto_target_net.load_state_dict(self.proto_policy_net.state_dict())
            
            # Complete execution record
            execution["end_time"] = time.time()
            execution["execution_time"] = execution["end_time"] - execution["start_time"]
            self.task_history.append(execution)
            
            return predictions
            
        except Exception as e:
            traceback_str = traceback.format_exc()
            execution["error"] = str(e)
            execution["traceback"] = traceback_str
            execution["end_time"] = time.time()
            execution["execution_time"] = execution["end_time"] - execution["start_time"]
            self.task_history.append(execution)
            
            print(f"Error solving task: {e}")
            print(traceback_str)
            return self._fallback_solution(task)
    
    def get_best_module_sequence(self, task):
        """
        Get the best module sequence for a task based on prototype similarity
        
        Args:
            task: Task object
            
        Returns:
            List of module names
        """
        # Extract features and compute embedding
        features = self.feature_extractor(task)
        embedding = self.embedding_network(features)
        
        # Initialize empty sequence
        module_sequence = []
        
        # Find similar tasks in prototype memory
        if self.prototype_memory:
            # Compute similarities to stored task embeddings
            similarities = []
            for entry in self.prototype_memory:
                if entry["performance"] is not None:
                    stored_embedding = entry["embedding"]
                    similarity = torch.cosine_similarity(embedding, stored_embedding, dim=1).item()
                    similarities.append((similarity, entry["sequence"], entry["performance"]))
            
            # Sort by similarity and performance
            similarities.sort(key=lambda x: (x[0], x[2]), reverse=True)
            
            # Get sequences from most similar tasks with high performance
            top_sequences = []
            for sim, seq, perf in similarities[:5]:
                if perf > 0.7:  # Only consider sequences that performed well
                    top_sequences.append((seq, sim * perf))  # Weight by both similarity and performance
            
            if top_sequences:
                # Use the best sequence
                module_sequence = top_sequences[0][0]
                return module_sequence
        
        # If no good matches in memory or empty memory, use prototype network
        with torch.no_grad():
            # Get Q-values from prototype network
            q_values, proto_weights, similarities = self.proto_policy_net(embedding)
            
            # Track sequence of modules
            prev_actions_tensor = torch.zeros(1, self.action_size, device=self.device)
            
            # Apply modules until max sequence length
            for step in range(self.config["max_sequence_length"]):
                if step == 0:
                    # First module selection
                    action = q_values.max(1)[1].item()
                else:
                    # For subsequent modules, decide whether to continue
                    continue_logits = self.sequence_net(embedding, prev_actions_tensor)
                    continue_prob = torch.softmax(continue_logits, dim=1)[0, 1].item()
                    
                    if continue_prob < 0.5:
                        # Stop sequence if model suggests stopping
                        break
                    
                    # Select next module (avoid repeating)
                    q_copy = q_values.clone()
                    for i in range(self.action_size):
                        if prev_actions_tensor[0, i] > 0:
                            q_copy[0, i] = -float('inf')
                    
                    action = q_copy.max(1)[1].item()
                
                # Add selected module to sequence
                module_name = self.module_names[action]
                module_sequence.append(module_name)
                
                # Update previous actions tensor
                prev_actions_tensor[0, action] = 1.0
        
        return module_sequence
    
    def _fallback_solution(self, task):
        """
        Generate a fallback solution when errors occur
        
        Args:
            task: Task object
            
        Returns:
            List of predictions (copy of inputs)
        """
        # Default to copying the input grids
        predictions = []
        for input_grid, _ in task.test_pairs:
            predictions.append(np.array(input_grid))
        return predictions
        
    def _capture_blackboard_state(self, blackboard):
        """
        Capture the current state of the blackboard
        
        Args:
            blackboard: Task blackboard
            
        Returns:
            Dictionary with blackboard state summary
        """
        state = {
            "predicates_count": 0,
            "transformations": {},
            "text_count": 0,
            "reasoning_steps": 0,
            "confidence_scores": {}
        }
        
        # Adapt this to your blackboard structure
        if hasattr(blackboard, 'logical_predicates'):
            state["predicates_count"] = len(blackboard.logical_predicates)
        elif hasattr(blackboard, 'knowledge_base'):
            state["predicates_count"] = sum(1 for k in blackboard.knowledge_base.keys() if 'predicate' in k.lower())
        
        # Count transformations
        if hasattr(blackboard, 'transformations'):
            if isinstance(blackboard.transformations, dict):
                for source, transforms in blackboard.transformations.items():
                    state["transformations"][source] = len(transforms)
            else:
                state["transformations"]["generic"] = len(blackboard.transformations)
        elif hasattr(blackboard, 'knowledge_base'):
            sources = {}
            for k, v in zip(blackboard.knowledge_base.keys(), blackboard.knowledge_sources.values()):
                if 'transform' in k.lower():
                    if v not in sources:
                        sources[v] = 0
                    sources[v] += 1
            state["transformations"] = sources
        
        # Count text entries
        if hasattr(blackboard, 'textual_data'):
            state["text_count"] = len(blackboard.textual_data)
        
        # Count reasoning steps
        if hasattr(blackboard, 'reasoning_history'):
            state["reasoning_steps"] = len(blackboard.reasoning_history)
        
        # Get confidence scores
        if hasattr(blackboard, 'confidence_scores'):
            state["confidence_scores"] = copy.deepcopy(blackboard.confidence_scores)
        
        return state
    
    def _compute_blackboard_delta(self, pre_state, post_state):
        """
        Compute changes in blackboard state
        
        Args:
            pre_state: State before module execution
            post_state: State after module execution
            
        Returns:
            Dictionary with changes
        """
        delta = {
            "predicates_added": post_state["predicates_count"] - pre_state["predicates_count"],
            "transformations_delta": {},
            "text_added": post_state["text_count"] - pre_state["text_count"],
            "reasoning_steps_added": post_state["reasoning_steps"] - pre_state["reasoning_steps"],
            "confidence_changes": {}
        }
        
        # Calculate transformation changes by source
        for source in set(list(pre_state["transformations"].keys()) + list(post_state["transformations"].keys())):
            pre_count = pre_state["transformations"].get(source, 0)
            post_count = post_state["transformations"].get(source, 0)
            if pre_count != post_count:
                delta["transformations_delta"][source] = post_count - pre_count
        
        # Calculate confidence score changes
        pre_scores = pre_state.get("confidence_scores", {})
        post_scores = post_state.get("confidence_scores", {})
        
        for module in set(list(pre_scores.keys()) + list(post_scores.keys())):
            pre_score = 0
            post_score = 0
            
            if module in pre_scores:
                if isinstance(pre_scores[module], list):
                    pre_score = sum(pre_scores[module]) / len(pre_scores[module]) if pre_scores[module] else 0
                else:
                    pre_score = pre_scores[module]
            
            if module in post_scores:
                if isinstance(post_scores[module], list):
                    post_score = sum(post_scores[module]) / len(post_scores[module]) if post_scores[module] else 0
                else:
                    post_score = post_scores[module]
            
            if pre_score != post_score:
                delta["confidence_changes"][module] = post_score - pre_score
        
        return delta
        
    def evaluate_predictions(self, predictions, task):
        """
        Evaluate prediction accuracy
        
        Args:
            predictions: List of predicted grids
            task: Task object with test pairs
            
        Returns:
            Tuple of (node_accuracy, grid_accuracy)
        """
        total_node_accuracy = 0.0
        grid_correct = 0
        
        for i, prediction in enumerate(predictions):
            if i < len(task.test_pairs):  # Ensure valid test pair index
                # Get target grid
                _, target_grid = task.test_pairs[i]
                target_np = np.array(target_grid)
                
                # Ensure prediction has the same shape as target
                if prediction.shape != target_np.shape:
                    # Resize to smaller of the two shapes
                    min_rows = min(prediction.shape[0], target_np.shape[0])
                    min_cols = min(prediction.shape[1], target_np.shape[1])
                    prediction = prediction[:min_rows, :min_cols]
                    target_np = target_np[:min_rows, :min_cols]
                
                # Calculate node-level accuracy (percentage of matching cells)
                matches = (prediction == target_np).sum()
                total = target_np.size
                node_accuracy = matches / total if total > 0 else 0.0
                total_node_accuracy += node_accuracy
                
                # Calculate grid-level accuracy (1 if grid is completely correct, 0 otherwise)
                if matches == total:
                    grid_correct += 1
        
        # Calculate average accuracies
        node_accuracy = total_node_accuracy / len(predictions) if predictions else 0.0
        grid_accuracy = grid_correct / len(predictions) if predictions else 0.0
        
        return node_accuracy, grid_accuracy
    
    def update_prototypes_from_memory(self):
        """
        Update prototype vectors using successful task solutions in memory
        """
        # Need enough memory entries
        if len(self.prototype_memory) < 10:
            return
            
        # Filter for successful sequences (accuracy > 0.7)
        successful_entries = [entry for entry in self.prototype_memory 
                             if entry["performance"] is not None and entry["performance"] > 0.7]
        
        if not successful_entries:
            return
            
        # Stack embeddings and create labels
        embeddings = torch.cat([entry["embedding"] for entry in successful_entries])
        
        # Simple k-means like update
        with torch.no_grad():
            # For each prototype
            for p in range(self.proto_policy_net.num_prototypes):
                prototype = self.proto_policy_net.prototypes[p]
                
                # Find similar embeddings
                similarities = torch.cosine_similarity(
                    prototype.unsqueeze(0), 
                    embeddings, 
                    dim=1
                )
                
                # Get indices of top similar embeddings
                _, indices = similarities.topk(min(5, len(successful_entries)))
                
                if len(indices) > 0:
                    # Update prototype as weighted average
                    weighted_sum = torch.zeros_like(prototype)
                    weight_sum = 0.0
                    
                    for idx in indices:
                        entry = successful_entries[idx]
                        weight = entry["performance"]  # Use performance as weight
                        weighted_sum += entry["embedding"].squeeze(0) * weight
                        weight_sum += weight
                    
                    if weight_sum > 0:
                        # Update prototype (partial update)
                        update_rate = self.config["prototype_update_rate"]
                        new_prototype = weighted_sum / weight_sum
                        updated_prototype = (1 - update_rate) * prototype + update_rate * new_prototype
                        
                        # Normalize to unit length
                        self.proto_policy_net.prototypes.data[p] = nn.functional.normalize(
                            updated_prototype, p=2, dim=0
                        )
            
            # Update corresponding actions for prototypes
            for p in range(self.proto_policy_net.num_prototypes):
                prototype = self.proto_policy_net.prototypes[p]
                
                # Find similar embeddings
                similarities = torch.cosine_similarity(
                    prototype.unsqueeze(0), 
                    embeddings, 
                    dim=1
                )
                
                # Get indices of top similar embeddings
                _, indices = similarities.topk(min(5, len(successful_entries)))
                
                if len(indices) > 0:
                    # Create action vector from successful sequences
                    action_vectors = []
                    
                    for idx in indices:
                        entry = successful_entries[idx]
                        # Create one-hot action vector for first action in sequence
                        if entry["sequence"]:
                            first_action = self.module_names.index(entry["sequence"][0])
                            action_vector = torch.zeros(self.action_size, device=self.device)
                            action_vector[first_action] = 1.0
                            action_vectors.append((action_vector, entry["performance"]))
                    
                    if action_vectors:
                        # Compute weighted average of action vectors
                        weighted_sum = torch.zeros(self.action_size, device=self.device)
                        weight_sum = 0.0
                        
                        for action_vec, weight in action_vectors:
                            weighted_sum += action_vec * weight
                            weight_sum += weight
                        
                        if weight_sum > 0:
                            # Update action mapping (partial update)
                            update_rate = self.config["prototype_update_rate"]
                            self.proto_policy_net.prototype_actions.data[p] = (
                                (1 - update_rate) * self.proto_policy_net.prototype_actions.data[p] + 
                                update_rate * (weighted_sum / weight_sum)
                            )
        
        # Copy to target network
        self.proto_target_net.load_state_dict(self.proto_policy_net.state_dict())
    
    def visualize_prototypes(self, save_path=None):
        """
        Visualize prototypes and their action preferences
        
        Args:
            save_path: Path to save visualization
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Need enough memory entries
        if len(self.prototype_memory) < 5:
            print(f"Not enough memory entries for visualization (have {len(self.prototype_memory)}, need at least 5)")
            return
        
        # Extract embeddings and performance
        embeddings = torch.cat([entry["embedding"] for entry in self.prototype_memory])
        embeddings_np = embeddings.detach().cpu().numpy()
        
        # Check dimensions
        if embeddings_np.ndim == 1:
            # This is a single vector, reshape it
            embeddings_np = embeddings_np.reshape(1, -1)
        
        performances = np.array([entry["performance"] if entry["performance"] is not None else 0.0 
                               for entry in self.prototype_memory])
        
        # Get prototype embeddings
        prototypes = self.proto_policy_net.prototypes.detach().cpu().numpy()
        prototype_actions = self.proto_policy_net.prototype_actions.detach().cpu().numpy()
        
        try:
            # Check if we have enough samples for PCA
            n_samples = embeddings_np.shape[0]
            
            if n_samples < 2:
                print(f"Cannot perform PCA with only {n_samples} sample(s). Need at least 2 samples.")
                
                # Create a simple visualization without PCA
                plt.figure(figsize=(10, 8))
                
                # Just plot the prototypes in a grid
                n_prototypes = prototypes.shape[0]
                grid_size = int(np.ceil(np.sqrt(n_prototypes)))
                
                # Create a grid layout
                x_positions = np.repeat(np.arange(grid_size), grid_size)[:n_prototypes]
                y_positions = np.tile(np.arange(grid_size), grid_size)[:n_prototypes]
                
                # Plot prototypes
                plt.scatter(x_positions, y_positions, c='red', marker='*', s=200, label='Prototypes')
                
                # Annotate prototypes with their associated action sequences
                for i, (x, y) in enumerate(zip(x_positions, y_positions)):
                    # Find the most similar stored embedding sequence for each prototype
                    prototype_tensor = self.proto_policy_net.prototypes[i].unsqueeze(0)
                    best_match = None
                    best_similarity = -float('inf')
                
                    # Iterate over prototype_memory to find the closest match
                    for entry in self.prototype_memory:
                        stored_embedding = entry["embedding"]
                        similarity = torch.cosine_similarity(prototype_tensor, stored_embedding, dim=1).item()
                        if similarity > best_similarity and entry["sequence"]:
                            best_similarity = similarity
                            best_match = entry["sequence"]
                
                    # Use the matched action sequence if found; else default to 'unknown'
                    if best_match:
                        action_sequence_label = " → ".join(best_match)
                    else:
                        action_sequence_label = "unknown"
                
                    plt.annotate(
                        f"{i}: {action_sequence_label}",
                        (x, y),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
                    )
                
                plt.title('Prototype Module Preferences (Grid Layout)')
                plt.xlim(-0.5, grid_size - 0.5)
                plt.ylim(-0.5, grid_size - 0.5)
                plt.axis('off')  # Turn off axes for cleaner look
            else:
                # We have enough samples for PCA
                from sklearn.decomposition import PCA
                
                # Determine maximum number of components
                max_components = min(n_samples, embeddings_np.shape[1], 2)
                
                # Project to 2D (or fewer) for visualization
                pca = PCA(n_components=max_components)
                embeddings_proj = pca.fit_transform(embeddings_np)
                prototypes_proj = pca.transform(prototypes)
                
                # If we only got 1 component, add a zero column
                if max_components == 1:
                    embeddings_proj = np.column_stack([embeddings_proj, np.zeros_like(embeddings_proj)])
                    prototypes_proj = np.column_stack([prototypes_proj, np.zeros_like(prototypes_proj)])
                
                # Create the plot
                plt.figure(figsize=(12, 10))
                
                # Plot task embeddings
                plt.scatter(
                    embeddings_proj[:, 0], 
                    embeddings_proj[:, 1], 
                    c=performances,
                    cmap='viridis',
                    alpha=0.6,
                    s=50,
                    vmin=0.0,
                    vmax=1.0,
                    label='Task Embeddings'
                )
                plt.colorbar(label='Performance')
                
                # Plot prototype embeddings
                plt.scatter(
                    prototypes_proj[:, 0],
                    prototypes_proj[:, 1],
                    c='red',
                    marker='*',
                    s=200,
                    label='Prototypes'
                )
                
                # Annotate prototypes with their associated action sequences
                for i, (x, y) in enumerate(zip(prototypes_proj[:, 0], prototypes_proj[:, 1])):
                    # Find the most similar stored embedding sequence for each prototype
                    prototype_tensor = self.proto_policy_net.prototypes[i].unsqueeze(0)
                    best_match = None
                    best_similarity = -float('inf')
                
                    # Iterate over prototype_memory to find the closest match
                    for entry in self.prototype_memory:
                        stored_embedding = entry["embedding"]
                        similarity = torch.cosine_similarity(prototype_tensor, stored_embedding, dim=1).item()
                        if similarity > best_similarity and entry["sequence"]:
                            best_similarity = similarity
                            best_match = entry["sequence"]
                
                    # Use the matched action sequence if found; else default to 'unknown'
                    if best_match:
                        action_sequence_label = " → ".join(best_match)
                    else:
                        action_sequence_label = "unknown"
                
                    plt.annotate(
                        f"{i}: {action_sequence_label}",
                        (x, y),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
                    )
                
                plt.title('Prototype and Task Embeddings')
                plt.xlabel(f'Component 1 ({pca.explained_variance_ratio_[0]:.2%})' if max_components > 0 else 'Component 1')
                if max_components > 1:
                    plt.ylabel(f'Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
                else:
                    plt.ylabel('Component 2')
                plt.legend()
            
            plt.tight_layout()
            
            if save_path:
                # Ensure directory exists
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                
            plt.show()
            plt.close()  # Close to avoid displaying the plot in notebooks
            
        except Exception as e:
            print(f"Error in visualizing prototypes: {e}")
            import traceback
            traceback.print_exc()
        
    def save_model(self, path):
        """
        Save the complete model state
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state_dict = {
            'embedding_network': self.embedding_network.state_dict(),
            'proto_policy_net': self.proto_policy_net.state_dict(),
            'proto_target_net': self.proto_target_net.state_dict(),
            'sequence_net': self.sequence_net.state_dict(),
            'module_names': self.module_names,
            'config': self.config,
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'module_performance': self.module_performance,
            'sequence_performance': self.sequence_performance,
            'prototype_memory': [
                {
                    'embedding': entry['embedding'].cpu(),
                    'sequence': entry['sequence'],
                    'performance': entry['performance']
                }
                for entry in self.prototype_memory
            ]
        }
        
        torch.save(state_dict, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load the complete model state
        
        Args:
            path: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(path):
            print(f"Model file not found: {path}")
            return False
        
        try:
            state_dict = torch.load(path, map_location=self.device)
            
            # Check if module names match
            stored_module_names = state_dict.get('module_names', [])
            if set(stored_module_names) != set(self.module_names):
                print(f"Warning: Loaded model has different modules. Expected {self.module_names}, got {stored_module_names}")
            
            # Update configuration if present
            if 'config' in state_dict:
                # Only update config parameters that don't affect network architecture
                safe_params = [
                    'gamma', 'learning_rate', 'update_target_every', 
                    'epsilon_start', 'epsilon_end', 'epsilon_decay',
                    'temperature', 'prototype_update_rate'
                ]
                for param in safe_params:
                    if param in state_dict['config']:
                        self.config[param] = state_dict['config'][param]
            
            # Load network states
            self.embedding_network.load_state_dict(state_dict['embedding_network'])
            self.proto_policy_net.load_state_dict(state_dict['proto_policy_net'])
            self.proto_target_net.load_state_dict(state_dict['proto_target_net'])
            self.sequence_net.load_state_dict(state_dict['sequence_net'])
            
            # Load training state
            self.epsilon = state_dict.get('epsilon', self.config['epsilon_start'])
            self.episode_count = state_dict.get('episode_count', 0)
            
            # Load performance tracking
            if 'module_performance' in state_dict:
                # Merge performance data for existing modules
                for module_name, perf in state_dict['module_performance'].items():
                    if module_name in self.module_names:
                        self.module_performance[module_name] = perf
            
            if 'sequence_performance' in state_dict:
                self.sequence_performance = state_dict['sequence_performance']
            
            # Load prototype memory
            if 'prototype_memory' in state_dict:
                self.prototype_memory = [
                    {
                        'embedding': entry['embedding'].to(self.device),
                        'sequence': entry['sequence'],
                        'performance': entry['performance']
                    }
                    for entry in state_dict['prototype_memory']
                ]
            
            print(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            return False