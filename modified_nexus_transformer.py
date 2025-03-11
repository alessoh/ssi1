import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm
import time
import sys

# Import the synthetic data generator
from synthetic_medical_data import generate_synthetic_patients

# ===============================
# Neural Components
# ===============================

class KnowledgeAwareAttention(nn.Module):
    """
    Attention mechanism that incorporates knowledge graph information
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Standard attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Knowledge-specific projection
        self.kg_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, kg_info=None, attention_mask=None):
        batch_size = query.shape[0]
        
        # Project queries, keys and values
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Incorporate knowledge graph information if provided
        if kg_info is not None:
            kg_embed = self.kg_proj(kg_info).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            kg_influence = torch.matmul(q, kg_embed.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # Add knowledge influence to attention scores
            scores = scores + kg_influence
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class SymbolicConstraintLayer(nn.Module):
    """
    Layer that applies symbolic constraints to neural representations
    """
    def __init__(self, embed_dim, num_constraints):
        super().__init__()
        self.constraint_weights = nn.Parameter(torch.randn(num_constraints, embed_dim))
        self.constraint_bias = nn.Parameter(torch.zeros(num_constraints))
        
    def forward(self, x):
        # Calculate constraint satisfaction scores
        # Reshape x to 2D for linear operation if needed
        batch_size, seq_len, embed_dim = x.shape
        x_reshaped = x.reshape(-1, embed_dim)
        
        scores = F.linear(x_reshaped, self.constraint_weights, self.constraint_bias)
        
        # Apply sigmoid to get constraint satisfaction probability
        satisfaction = torch.sigmoid(scores)
        
        # Reshape scores back to match input dimensions
        satisfaction = satisfaction.reshape(batch_size, seq_len, -1)
        
        # Create a soft mask based on constraint satisfaction
        # This allows the gradient to flow, unlike a hard binary mask
        # We need to ensure mask has same dimensions as x
        mask = satisfaction.mean(dim=2, keepdim=True).expand_as(x)
        
        # Apply mask to input (allows information flow proportional to constraint satisfaction)
        constrained_x = x * mask
        
        return constrained_x, satisfaction


class NeuralModule(nn.Module):
    """
    Neural foundation of the NEXUS-Transformer
    """
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, num_constraints, num_classes, dropout=0.1):
        super().__init__()
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))  # Single position for input
        
        # Knowledge-aware attention layers
        self.attention_layers = nn.ModuleList([
            KnowledgeAwareAttention(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Feed-forward layers
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.GELU(),
                nn.Linear(4 * embed_dim, embed_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.layer_norms2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        # Symbolic constraint layer
        self.symbolic_constraint = SymbolicConstraintLayer(embed_dim, num_constraints)
        
        # Output projections for each condition
        self.output_projs = nn.ModuleDict({
            name: nn.Linear(embed_dim, 1) for name in num_classes
        })
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_features, knowledge_embeddings=None, attention_mask=None):
        batch_size = input_features.size(0)
        
        # Convert input features to embeddings
        x = self.input_embedding(input_features).unsqueeze(1)  # Add sequence dimension
        
        # Add position embeddings
        x = x + self.position_embedding
        
        x = self.dropout(x)
        
        # Apply transformer layers with knowledge-aware attention
        all_attentions = []
        for i in range(len(self.attention_layers)):
            # Self-attention with knowledge incorporation
            residual = x
            x = self.layer_norms1[i](x)
            x, attention = self.attention_layers[i](x, x, x, knowledge_embeddings, attention_mask)
            x = self.dropout(x)
            x = residual + x
            
            # Feed-forward
            residual = x
            x = self.layer_norms2[i](x)
            x = self.ff_layers[i](x)
            x = residual + x
            
            all_attentions.append(attention)
        
        # Apply symbolic constraints
        constrained_x, constraint_satisfaction = self.symbolic_constraint(x)
        
        # Project to outputs for each condition/need
        logits = {}
        for name, proj in self.output_projs.items():
            # Squeeze to remove sequence dimension
            logits[name] = proj(constrained_x).squeeze(1)
        
        return {
            'logits': logits,
            'hidden_states': constrained_x,
            'attention_weights': all_attentions,
            'constraint_satisfaction': constraint_satisfaction
        }


# ===============================
# Symbolic Components
# ===============================

class KnowledgeGraph:
    """
    Simple knowledge graph implementation with basic reasoning capabilities
    """
    def __init__(self):
        # Initialize empty graph
        self.entities = {}  # entity_id -> {name, attributes, embedding}
        self.relations = {}  # relation_id -> {source, target, type, weight}
        self.rules = {}  # rule_id -> {premise, conclusion, confidence}
        
    def add_entity(self, entity_id, name, attributes=None, embedding=None):
        """Add an entity to the knowledge graph"""
        self.entities[entity_id] = {
            'name': name,
            'attributes': attributes or {},
            'embedding': embedding
        }
        return self
        
    def add_relation(self, relation_id, source_id, target_id, relation_type, weight=1.0):
        """Add a relation between two entities"""
        self.relations[relation_id] = {
            'source': source_id,
            'target': target_id,
            'type': relation_type,
            'weight': weight
        }
        return self
        
    def add_rule(self, rule_id, premise, conclusion, confidence=1.0):
        """Add a logical rule to the knowledge graph"""
        self.rules[rule_id] = {
            'premise': premise,  # List of entity IDs or patterns
            'conclusion': conclusion,  # Entity ID or pattern to infer
            'confidence': confidence
        }
        return self
        
    def get_entity_embedding(self, entity_id):
        """Get the embedding for an entity"""
        return self.entities.get(entity_id, {}).get('embedding')
        
    def get_related_entities(self, entity_id, relation_type=None):
        """Get entities related to a given entity"""
        related = []
        for rel_id, rel in self.relations.items():
            if rel['source'] == entity_id and (relation_type is None or rel['type'] == relation_type):
                related.append(rel['target'])
            if rel['target'] == entity_id and (relation_type is None or rel['type'] == relation_type):
                related.append(rel['source'])
        return related
    
    def reason(self, active_entities):
        """
        Apply logical rules to infer new knowledge
        
        Args:
            active_entities: Set of currently active entity IDs
            
        Returns:
            Set of inferred entity IDs
        """
        inferred = set(active_entities)
        changed = True
        
        # Apply rules until no more changes
        while changed:
            changed = False
            for rule_id, rule in self.rules.items():
                # Check if premise is satisfied
                premise_satisfied = True
                for entity_pattern in rule['premise']:
                    if isinstance(entity_pattern, tuple):  # (entity_id, relation_type)
                        entity_id, relation_type = entity_pattern
                        if entity_id not in active_entities:
                            premise_satisfied = False
                            break
                        
                        # Check if entity has related entities of the specified type
                        related = self.get_related_entities(entity_id, relation_type)
                        if not any(rel in active_entities for rel in related):
                            premise_satisfied = False
                            break
                    else:  # Simple entity_id
                        if entity_pattern not in active_entities:
                            premise_satisfied = False
                            break
                
                # If premise is satisfied, add conclusion to inferred entities
                if premise_satisfied:
                    if isinstance(rule['conclusion'], tuple):
                        entity_id, relation_type = rule['conclusion']
                        related = self.get_related_entities(entity_id, relation_type)
                        for rel in related:
                            if rel not in inferred:
                                inferred.add(rel)
                                changed = True
                    else:
                        if rule['conclusion'] not in inferred:
                            inferred.add(rule['conclusion'])
                            changed = True
        
        return inferred


class SymbolicModule:
    """
    Symbolic component of the NEXUS-Transformer
    """
    def __init__(self, knowledge_graph, symbol_space_size, embed_dim):
        self.kg = knowledge_graph
        self.symbol_space_size = symbol_space_size
        self.embed_dim = embed_dim
        
        # Create embedding matrix for symbols
        # In a real implementation, these would be learned or derived from knowledge
        self.symbol_embeddings = torch.randn(symbol_space_size, embed_dim)
        
    def get_embedding(self, symbol_ids):
        """Get embeddings for symbols"""
        return self.symbol_embeddings[symbol_ids]
        
    def activate_symbols(self, features, threshold=0.5):
        """
        Convert input features to symbolic activations
        
        Args:
            features: Tensor of shape [batch_size, feature_dim]
            threshold: Activation threshold
            
        Returns:
            Tensor of shape [batch_size, symbol_space_size] with 1s for active symbols
        """
        batch_size = features.shape[0]
        
        # Simple linear mapping from features to symbol space with some randomness
        similarity = torch.matmul(features, self.symbol_embeddings.t())
        
        # Apply threshold to get active symbols
        active_symbols = (similarity > threshold).float()
        
        return active_symbols, similarity
        
    def reason(self, active_symbols):
        """
        Apply symbolic reasoning to active symbols
        
        Args:
            active_symbols: Tensor of shape [batch_size, symbol_space_size]
            
        Returns:
            Tensor of shape [batch_size, symbol_space_size] with inferred symbols
        """
        batch_size = active_symbols.shape[0]
        inferred_symbols = active_symbols.clone()
        
        # Apply reasoning separately for each example in the batch
        for i in range(batch_size):
            active_ids = torch.nonzero(active_symbols[i]).squeeze(-1).tolist()
            inferred_ids = self.kg.reason(active_ids)
            for symbol_id in inferred_ids:
                if symbol_id < self.symbol_space_size:
                    inferred_symbols[i, symbol_id] = 1.0
        
        return inferred_symbols


# ===============================
# Neural-Symbolic Interface
# ===============================

class NeuralSymbolicInterface:
    """
    Interface for bidirectional translation between neural and symbolic representations
    """
    def __init__(self, neural_module, symbolic_module):
        self.neural = neural_module
        self.symbolic = symbolic_module
        
    def neural_to_symbolic(self, features, threshold=0.5):
        """Convert neural representations to symbolic activations"""
        return self.symbolic.activate_symbols(features, threshold)
        
    def symbolic_to_neural(self, symbolic_activations):
        """Convert symbolic activations to neural representations"""
        # Compute weighted sum of symbol embeddings
        batch_size = symbolic_activations.shape[0]
        expanded_activations = symbolic_activations.unsqueeze(-1)  # [batch_size, symbol_space_size, 1]
        expanded_embeddings = self.symbolic.symbol_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, symbol_space_size, embed_dim]
        
        weighted_embeddings = expanded_activations * expanded_embeddings
        neural_repr = weighted_embeddings.sum(dim=1)  # [batch_size, embed_dim]
        
        # Expand to match the neural module's expected input
        neural_repr = neural_repr.unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        return neural_repr


# ===============================
# Metacognitive Controller
# ===============================

class MetacognitiveController:
    """
    Controller that determines when to use neural vs. symbolic processing
    """
    def __init__(self, neural_threshold=0.7, symbolic_threshold=0.8):
        self.neural_threshold = neural_threshold
        self.symbolic_threshold = symbolic_threshold
        
    def decide_strategy(self, neural_confidence, symbolic_confidence, risk_level='medium'):
        """
        Decide which processing strategy to use
        
        Args:
            neural_confidence: Confidence of neural prediction
            symbolic_confidence: Confidence of symbolic prediction
            risk_level: Risk level of the current task ('low', 'medium', 'high')
            
        Returns:
            Dictionary with strategy and weights
        """
        # Adjust thresholds based on risk level
        if risk_level == 'high':
            # In high-risk scenarios, favor symbolic reasoning
            neural_threshold = self.neural_threshold + 0.1
            symbolic_threshold = self.symbolic_threshold - 0.1
        elif risk_level == 'low':
            # In low-risk scenarios, favor neural processing
            neural_threshold = self.neural_threshold - 0.1
            symbolic_threshold = self.symbolic_threshold + 0.1
        else:
            neural_threshold = self.neural_threshold
            symbolic_threshold = self.symbolic_threshold
        
        # Determine strategy based on confidence levels
        if neural_confidence >= neural_threshold and symbolic_confidence < symbolic_threshold:
            return {
                'strategy': 'neural',
                'neural_weight': 1.0,
                'symbolic_weight': 0.0,
                'explanation': 'Using neural prediction due to high neural confidence'
            }
        elif symbolic_confidence >= symbolic_threshold and neural_confidence < neural_threshold:
            return {
                'strategy': 'symbolic',
                'neural_weight': 0.0,
                'symbolic_weight': 1.0,
                'explanation': 'Using symbolic reasoning due to high symbolic confidence'
            }
        else:
            # Use weighted combination
            total_confidence = neural_confidence + symbolic_confidence
            neural_weight = neural_confidence / total_confidence
            symbolic_weight = symbolic_confidence / total_confidence
            
            return {
                'strategy': 'hybrid',
                'neural_weight': neural_weight,
                'symbolic_weight': symbolic_weight,
                'explanation': 'Using weighted combination of neural and symbolic processing'
            }


# ===============================
# Explanation Generator
# ===============================

class ExplanationGenerator:
    """
    Generates human-understandable explanations for the model's decisions
    """
    def __init__(self, knowledge_graph, symbol_names):
        self.kg = knowledge_graph
        self.symbol_names = symbol_names
        
    def generate_explanation(self, active_symbols, inferred_symbols, strategy, level='simple'):
        """
        Generate explanation for the current prediction
        
        Args:
            active_symbols: Tensor of shape [batch_size, symbol_space_size]
            inferred_symbols: Tensor of shape [batch_size, symbol_space_size]
            strategy: Strategy dictionary from metacognitive controller
            level: Detail level ('simple', 'detailed', 'technical')
            
        Returns:
            List of explanation strings
        """
        batch_size = active_symbols.shape[0]
        explanations = []
        
        for i in range(batch_size):
            # Get active and inferred symbols
            active_ids = torch.nonzero(active_symbols[i]).squeeze(-1).tolist()
            inferred_ids = torch.nonzero(inferred_symbols[i]).squeeze(-1).tolist()
            newly_inferred = [idx for idx in inferred_ids if idx not in active_ids]
            
            # Convert to symbol names
            active_names = [self.symbol_names[idx] for idx in active_ids if idx < len(self.symbol_names)]
            inferred_names = [self.symbol_names[idx] for idx in newly_inferred if idx < len(self.symbol_names)]
            
            # Generate explanation based on level
            if level == 'simple':
                exp = f"Strategy: {strategy['strategy']}\n"
                exp += f"Identified concepts: {', '.join(active_names)}\n"
                if inferred_names:
                    exp += f"Inferred concepts: {', '.join(inferred_names)}"
                else:
                    exp += "No additional concepts inferred."
            elif level == 'detailed':
                exp = f"Strategy: {strategy['strategy']} (Neural weight: {strategy['neural_weight']:.2f}, Symbolic weight: {strategy['symbolic_weight']:.2f})\n"
                exp += f"Identified concepts: {', '.join(active_names)}\n"
                
                if inferred_names:
                    exp += f"Inferred concepts: {', '.join(inferred_names)}\n"
                    exp += "Reasoning path:\n"
                    
                    # Explain each inference (simplified)
                    for symbol in inferred_names:
                        try:
                            symbol_id = self.symbol_names.index(symbol)
                            exp += f"  - {symbol} was inferred because: "
                            
                            # Find rules that could have led to this inference
                            relevant_rules = []
                            for rule_id, rule in self.kg.rules.items():
                                if (isinstance(rule['conclusion'], int) and rule['conclusion'] == symbol_id) or \
                                   (isinstance(rule['conclusion'], tuple) and rule['conclusion'][0] == symbol_id):
                                    
                                    # Check if premises were active
                                    premise_symbols = []
                                    for premise in rule['premise']:
                                        if isinstance(premise, int) and premise < len(self.symbol_names):
                                            premise_symbols.append(self.symbol_names[premise])
                                        elif isinstance(premise, tuple) and premise[0] < len(self.symbol_names):
                                            premise_symbols.append(f"{self.symbol_names[premise[0]]} with relation {premise[1]}")
                                
                                relevant_rules.append(f"{' AND '.join(premise_symbols)}")
                        
                            if relevant_rules:
                                exp += f"{' OR '.join(relevant_rules)}"
                            else:
                                exp += "Unknown reasoning path"
                        except ValueError:
                            exp += f"  - {symbol} reasoning path not available"
                else:
                    exp += "No additional concepts inferred."
            else:  # technical
                # Technical level would include more details about activation values, etc.
                exp = f"Strategy: {strategy['strategy']} with weights [{strategy['neural_weight']:.4f}, {strategy['symbolic_weight']:.4f}]\n"
                exp += f"Active symbols (IDs): {active_ids}\n"
                exp += f"Inferred symbols (IDs): {newly_inferred}\n"
                exp += f"Active symbol names: {active_names}\n"
                exp += f"Inferred symbol names: {inferred_names}"
            
            explanations.append(exp)
        
        return explanations


# ===============================
# NEXUS-Transformer
# ===============================

class NEXUSTransformer:
    """
    Complete NEXUS-Transformer architecture for medical diagnosis
    """
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, num_constraints, 
                 conditions, symbol_names, neural_threshold=0.7, symbolic_threshold=0.8):
        # Initialize neural module
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.conditions = conditions
        self.symbol_names = symbol_names
        self.symbol_space_size = len(symbol_names)
        
        # Initialize neural module
        self.neural_module = NeuralModule(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_constraints=num_constraints,
            num_classes=conditions
        )
        
        # Initialize knowledge graph
        self.knowledge_graph = create_medical_knowledge_graph(symbol_names)
        
        # Initialize symbolic module
        self.symbolic_module = SymbolicModule(self.knowledge_graph, self.symbol_space_size, embed_dim)
        
        # Initialize neural-symbolic interface
        self.interface = NeuralSymbolicInterface(self.neural_module, self.symbolic_module)
        
        # Initialize metacognitive controller
        self.metacognitive = MetacognitiveController(neural_threshold, symbolic_threshold)
        
        # Initialize explanation generator
        self.explanation_generator = ExplanationGenerator(self.knowledge_graph, symbol_names)
        
    def process(self, features, risk_level='medium'):
        """
        Process patient features through the NEXUS-Transformer
        
        Args:
            features: Tensor of patient features [batch_size, input_dim]
            risk_level: Risk level of current patient ('low', 'medium', 'high')
            
        Returns:
            Dictionary with results and explanations
        """
        # Neural processing
        features_tensor = torch.FloatTensor(features)
        neural_output = self.neural_module(features_tensor)
        
        # Neural-to-symbolic translation
        active_symbols, symbol_similarities = self.interface.neural_to_symbolic(features_tensor)
        
        # Symbolic reasoning
        inferred_symbols = self.symbolic_module.reason(active_symbols)
        
        # Symbolic-to-neural translation
        symbolic_neural_repr = self.interface.symbolic_to_neural(inferred_symbols)
        
        # Compute confidences
        neural_confidence = torch.mean(torch.cat([
            torch.sigmoid(neural_output['logits'][cond]).max()
            for cond in self.conditions
        ])).item()
        
        symbolic_confidence = torch.mean(symbol_similarities.max(dim=1)[0]).item()
        
        # Metacognitive control
        strategy = self.metacognitive.decide_strategy(neural_confidence, symbolic_confidence, risk_level)
        
        # Generate predictions based on strategy
        final_predictions = {}
        if strategy['strategy'] == 'neural':
            # Use neural predictions
            for cond in self.conditions:
                final_predictions[cond] = torch.sigmoid(neural_output['logits'][cond])
        
        elif strategy['strategy'] == 'symbolic':
            # Use symbolic predictions
            for cond in self.conditions:
                cond_id = self.symbol_names.index(cond) if cond in self.symbol_names else -1
                if cond_id >= 0:
                    final_predictions[cond] = inferred_symbols[:, cond_id]
                else:
                    # Fallback to neural if condition not in symbolic
                    final_predictions[cond] = torch.sigmoid(neural_output['logits'][cond])
        
        else:  # hybrid
            neural_weight = strategy['neural_weight']
            symbolic_weight = strategy['symbolic_weight']
            
            # Weighted combination of neural and symbolic
            for cond in self.conditions:
                cond_id = self.symbol_names.index(cond) if cond in self.symbol_names else -1
                if cond_id >= 0:
                    neural_pred = torch.sigmoid(neural_output['logits'][cond])
                    symbolic_pred = inferred_symbols[:, cond_id]
                    
                    # Ensure tensor dimensions match
                    if symbolic_pred.dim() < neural_pred.dim():
                        symbolic_pred = symbolic_pred.unsqueeze(-1)
                    
                    final_predictions[cond] = neural_weight * neural_pred + symbolic_weight * symbolic_pred
                else:
                    # Fallback to neural if condition not in symbolic
                    final_predictions[cond] = torch.sigmoid(neural_output['logits'][cond])
        
        # Generate explanation
        explanations = self.explanation_generator.generate_explanation(
            active_symbols, inferred_symbols, strategy, level='simple'
        )
        
        return {
            'predictions': final_predictions,
            'active_symbols': active_symbols,
            'inferred_symbols': inferred_symbols,
            'strategy': strategy,
            'neural_confidence': neural_confidence,
            'symbolic_confidence': symbolic_confidence,
            'explanations': explanations
        }
    
    def predict(self, features, threshold=0.5, batch_size=64):
        """
        Make predictions on the given features
        
        Args:
            features: Array of patient features [n_samples, input_dim]
            threshold: Threshold for binary predictions
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with predicted conditions for each patient
        """
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        
        n_samples = features.shape[0]
        all_predictions = {cond: [] for cond in self.conditions}
        all_explanations = []
        all_strategies = []
        
        # Process in batches
        for i in range(0, n_samples, batch_size):
            batch_features = features[i:i+batch_size]
            
            # Process batch
            results = self.process(batch_features)
            
            # Collect predictions
            for cond in self.conditions:
                batch_preds = results['predictions'][cond].detach().squeeze().cpu().numpy()
                if isinstance(batch_preds, float):
                    batch_preds = np.array([batch_preds])
                all_predictions[cond].append(batch_preds)
            
            # Collect explanations
            all_explanations.extend(results['explanations'])
            all_strategies.append(results['strategy']['strategy'])
        
        # Concatenate batch results
        predictions = {cond: np.concatenate(all_predictions[cond]) > threshold for cond in self.conditions}
        
        return {
            'predictions': predictions,
            'explanations': all_explanations,
            'strategies': all_strategies
        }
    
    def evaluate(self, features, labels, threshold=0.5, batch_size=64):
        """
        Evaluate model performance on test data
        
        Args:
            features: Array of patient features [n_samples, input_dim]
            labels: Dictionary of ground truth labels for each condition
            threshold: Threshold for binary predictions
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with performance metrics
        """
        # Get predictions
        results = self.predict(features, threshold, batch_size)
        predictions = results['predictions']
        
        # Compute metrics
        metrics = {}
        for cond in self.conditions:
            if cond in labels:
                true_labels = labels[cond]
                pred_labels = predictions[cond].astype(int)
                
                metrics[cond] = {
                    'accuracy': accuracy_score(true_labels, pred_labels),
                    'f1_score': f1_score(true_labels, pred_labels),
                    'confusion_matrix': confusion_matrix(true_labels, pred_labels)
                }
        
        # Strategy distribution
        strategies = np.array(results['strategies'])
        strategy_counts = {
            'neural': sum(strategies == 'neural'),
            'symbolic': sum(strategies == 'symbolic'),
            'hybrid': sum(strategies == 'hybrid')
        }
        
        return {
            'condition_metrics': metrics,
            'strategy_distribution': strategy_counts
        }


# ===============================
# Helper Functions
# ===============================

def create_medical_knowledge_graph(symbol_names):
    """Create a knowledge graph for medical diagnosis"""
    # Create knowledge graph
    kg = KnowledgeGraph()
    
    # Get indices for symbols
    symbol_indices = {name: i for i, name in enumerate(symbol_names)}
    
    # Add entities
    for i, name in enumerate(symbol_names):
        kg.add_entity(i, name)
    
    # Add relations between symptoms and conditions
    relation_id = 0
    
    # Fever relations
    kg.add_relation(relation_id, symbol_indices['fever'], symbol_indices['flu'], 'symptom_of', 0.7); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['high_fever'], symbol_indices['flu'], 'symptom_of', 0.9); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['fever'], symbol_indices['covid'], 'symptom_of', 0.8); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['high_fever'], symbol_indices['covid'], 'symptom_of', 0.7); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['fever'], symbol_indices['pneumonia'], 'symptom_of', 0.9); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['high_fever'], symbol_indices['pneumonia'], 'symptom_of', 0.9); relation_id += 1
    
    # Cough relations
    kg.add_relation(relation_id, symbol_indices['cough'], symbol_indices['common_cold'], 'symptom_of', 0.8); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['cough'], symbol_indices['flu'], 'symptom_of', 0.7); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['cough'], symbol_indices['covid'], 'symptom_of', 0.9); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['cough'], symbol_indices['pneumonia'], 'symptom_of', 0.8); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['severe_cough'], symbol_indices['covid'], 'symptom_of', 0.8); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['severe_cough'], symbol_indices['pneumonia'], 'symptom_of', 0.9); relation_id += 1
    
    # Fatigue relations
    kg.add_relation(relation_id, symbol_indices['fatigue'], symbol_indices['common_cold'], 'symptom_of', 0.6); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['fatigue'], symbol_indices['flu'], 'symptom_of', 0.9); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['fatigue'], symbol_indices['covid'], 'symptom_of', 0.8); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['fatigue'], symbol_indices['pneumonia'], 'symptom_of', 0.9); relation_id += 1
    
    # Other symptom relations
    kg.add_relation(relation_id, symbol_indices['headache'], symbol_indices['common_cold'], 'symptom_of', 0.5); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['headache'], symbol_indices['flu'], 'symptom_of', 0.7); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['shortness_of_breath'], symbol_indices['covid'], 'symptom_of', 0.8); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['shortness_of_breath'], symbol_indices['pneumonia'], 'symptom_of', 0.9); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['sore_throat'], symbol_indices['common_cold'], 'symptom_of', 0.9); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['sore_throat'], symbol_indices['flu'], 'symptom_of', 0.5); relation_id += 1
    
    # Risk factors relations
    kg.add_relation(relation_id, symbol_indices['elderly'], symbol_indices['need_hospitalization'], 'risk_factor', 0.7); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['immunocompromised'], symbol_indices['need_hospitalization'], 'risk_factor', 0.8); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['elderly'], symbol_indices['need_ventilator'], 'risk_factor', 0.6); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['immunocompromised'], symbol_indices['need_ventilator'], 'risk_factor', 0.7); relation_id += 1
    
    # Treatment relations
    kg.add_relation(relation_id, symbol_indices['common_cold'], symbol_indices['need_rest'], 'requires', 0.9); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['flu'], symbol_indices['need_rest'], 'requires', 0.8); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['covid'], symbol_indices['need_rest'], 'requires', 0.7); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['flu'], symbol_indices['need_testing'], 'requires', 0.6); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['covid'], symbol_indices['need_testing'], 'requires', 0.9); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['pneumonia'], symbol_indices['need_testing'], 'requires', 0.8); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['covid'], symbol_indices['need_hospitalization'], 'requires', 0.5); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['pneumonia'], symbol_indices['need_hospitalization'], 'requires', 0.8); relation_id += 1
    kg.add_relation(relation_id, symbol_indices['pneumonia'], symbol_indices['need_ventilator'], 'requires', 0.7); relation_id += 1
    
    # Add logical rules
    rule_id = 0
    
    # Symptom rules
    kg.add_rule(rule_id, [symbol_indices['fever'], symbol_indices['cough'], symbol_indices['fatigue']], 
                symbol_indices['flu'], 0.8); rule_id += 1
    
    kg.add_rule(rule_id, [symbol_indices['cough'], symbol_indices['shortness_of_breath'], symbol_indices['fatigue']], 
                symbol_indices['covid'], 0.7); rule_id += 1
    
    kg.add_rule(rule_id, [symbol_indices['severe_cough'], symbol_indices['shortness_of_breath']], 
                symbol_indices['pneumonia'], 0.9); rule_id += 1
    
    kg.add_rule(rule_id, [symbol_indices['sore_throat'], symbol_indices['cough']], 
                symbol_indices['common_cold'], 0.8); rule_id += 1
    
    # Risk factor rules
    kg.add_rule(rule_id, [symbol_indices['elderly'], symbol_indices['covid']], 
                symbol_indices['need_hospitalization'], 0.8); rule_id += 1
    
    kg.add_rule(rule_id, [symbol_indices['immunocompromised'], symbol_indices['covid']], 
                symbol_indices['need_hospitalization'], 0.9); rule_id += 1
    
    kg.add_rule(rule_id, [symbol_indices['elderly'], symbol_indices['pneumonia']], 
                symbol_indices['need_ventilator'], 0.7); rule_id += 1
    
    kg.add_rule(rule_id, [symbol_indices['immunocompromised'], symbol_indices['pneumonia']], 
                symbol_indices['need_ventilator'], 0.8); rule_id += 1
    
    return kg

def visualize_results(metrics, model_name="NEXUS-Transformer"):
    """
    Visualize evaluation results
    
    Args:
        metrics: Dictionary with performance metrics
        model_name: Name of the model
    """
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Performance metrics
    ax1 = fig.add_subplot(2, 2, 1)
    condition_metrics = metrics['condition_metrics']
    conditions = list(condition_metrics.keys())
    accuracies = [condition_metrics[cond]['accuracy'] for cond in conditions]
    f1_scores = [condition_metrics[cond]['f1_score'] for cond in conditions]
    
    x = np.arange(len(conditions))
    width = 0.35
    
    ax1.bar(x - width/2, accuracies, width, label='Accuracy')
    ax1.bar(x + width/2, f1_scores, width, label='F1 Score')
    
    ax1.set_title(f'{model_name} Performance Metrics')
    ax1.set_xlabel('Condition')
    ax1.set_ylabel('Score')
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Strategy distribution
    ax2 = fig.add_subplot(2, 2, 2)
    strategy_dist = metrics['strategy_distribution']
    strategies = list(strategy_dist.keys())
    counts = [strategy_dist[s] for s in strategies]
    
    ax2.bar(strategies, counts)
    ax2.set_title('Strategy Distribution')
    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Count')
    ax2.grid(alpha=0.3)
    
    # 3. Confusion matrices
    for i, cond in enumerate(conditions[:4]):  # Show max 4 confusion matrices
        if i >= 4:
            break
            
        ax = fig.add_subplot(2, 4, i+5)
        cm = condition_metrics[cond]['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix: {cond}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_results.png")
    plt.show()

# ===============================
# Main Script for 1000 Patients
# ===============================

def run_nexus_experiment(n_patients=1000, batch_size=64):
    """
    Run NEXUS-Transformer experiment with synthetic patient data
    """
    print("Generating synthetic medical data...")
    train_data, test_data = generate_synthetic_patients(n_patients)
    print(f"Generated {len(train_data)} training patients and {len(test_data)} test patients")
    
    # Define symbols and conditions
    symptoms = ['fever', 'high_fever', 'cough', 'severe_cough', 'fatigue', 
                'headache', 'shortness_of_breath', 'sore_throat']
    
    risk_factors = ['elderly', 'immunocompromised', 'child']
    
    conditions = ['common_cold', 'flu', 'covid', 'pneumonia', 'allergies',
                 'need_rest', 'need_testing', 'need_hospitalization', 'need_ventilator']
    
    symbol_names = symptoms + risk_factors + conditions
    
    # Extract features and labels
    test_features = np.stack(test_data['symptom_vector'].values)
    
    test_labels = {}
    for cond in conditions:
        test_labels[cond] = test_data[cond].values
    
    # Initialize NEXUS-Transformer
    print("\nInitializing NEXUS-Transformer...")
    nexus = NEXUSTransformer(
        input_dim=len(symptoms + risk_factors),
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        num_constraints=5,
        conditions=conditions,
        symbol_names=symbol_names,
        neural_threshold=0.6,  # Adjusted thresholds
        symbolic_threshold=0.5
    )
    
    # Train neural component (simplified)
    print("\nTraining neural component...")
    optimizer = torch.optim.Adam(nexus.neural_module.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    train_features = np.stack(train_data['symptom_vector'].values)
    train_labels = {cond: train_data[cond].values for cond in conditions}
    
    n_epochs = 5
    for epoch in range(n_epochs):
        epoch_loss = 0
        
        # Process in batches
        for i in range(0, len(train_features), batch_size):
            batch_features = torch.FloatTensor(train_features[i:i+batch_size])
            batch_labels = {cond: torch.FloatTensor(train_labels[cond][i:i+batch_size]) for cond in conditions}
            
            # Forward pass
            outputs = nexus.neural_module(batch_features)
            
            # Calculate loss
            loss = 0
            for cond in conditions:
                loss += criterion(outputs['logits'][cond].squeeze(), batch_labels[cond])
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(train_features):.4f}")
    
    # Evaluate on test data
    print("\nEvaluating NEXUS-Transformer on test data...")
    metrics = nexus.evaluate(test_features, test_labels, batch_size=batch_size)
    
    # Display sample explanations
    print("\nSample explanations:")
    sample_results = nexus.process(torch.FloatTensor(test_features[:5]))
    for i, exp in enumerate(sample_results['explanations']):
        print(f"\nPatient {i+1}:")
        print(exp)
    
    # Visualize results
    visualize_results(metrics)
    
    return nexus, test_data, metrics

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run experiment
    nexus, test_data, metrics = run_nexus_experiment(1000)
    
    # Print summary results
    print("\nPerformance Summary:")
    for cond, metric in metrics['condition_metrics'].items():
        print(f"{cond}: Accuracy={metric['accuracy']:.4f}, F1-Score={metric['f1_score']:.4f}")
    
    print("\nStrategy Distribution:")
    for strategy, count in metrics['strategy_distribution'].items():
        print(f"{strategy}: {count} patients")