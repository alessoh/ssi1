import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

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
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, num_constraints, dropout=0.1):
        super().__init__()
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, 100, embed_dim))  # Maximum sequence length of 100
        
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
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, knowledge_embeddings=None, attention_mask=None):
        # Get sequence length
        seq_length = input_ids.size(1)
        
        # Compute embeddings
        embeddings = self.token_embedding(input_ids)
        
        # Add position embeddings
        position_embeddings = self.position_embedding[:, :seq_length, :]
        x = embeddings + position_embeddings
        
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
        
        # Project to vocabulary
        logits = self.output_proj(constrained_x)
        
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
            'premise': premise,  # List of relation IDs or patterns
            'conclusion': conclusion,  # Relation ID or pattern to infer
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
        
    def activate_symbols(self, neural_repr, threshold=0.5):
        """
        Convert neural representations to symbolic activations
        
        Args:
            neural_repr: Tensor of shape [batch_size, seq_len, embed_dim]
            threshold: Activation threshold
            
        Returns:
            Tensor of shape [batch_size, symbol_space_size] with 1s for active symbols
        """
        # Simplified approach - just compute similarity to each symbol embedding
        batch_size, seq_len, _ = neural_repr.shape
        
        # Use only the first token representation for simplicity
        first_token = neural_repr[:, 0, :]  # [batch_size, embed_dim]
        
        # Compute similarity to each symbol embedding
        similarity = torch.matmul(first_token, self.symbol_embeddings.T)  # [batch_size, symbol_space_size]
        
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
        
    def neural_to_symbolic(self, neural_repr, threshold=0.5):
        """Convert neural representations to symbolic activations"""
        return self.symbolic.activate_symbols(neural_repr, threshold)
        
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
            active_names = [self.symbol_names[idx] for idx in active_ids]
            inferred_names = [self.symbol_names[idx] for idx in newly_inferred]
            
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
                                    if isinstance(premise, int):
                                        premise_symbols.append(self.symbol_names[premise])
                                    elif isinstance(premise, tuple):
                                        premise_symbols.append(f"{self.symbol_names[premise[0]]} with relation {premise[1]}")
                                
                                relevant_rules.append(f"{' AND '.join(premise_symbols)}")
                        
                        if relevant_rules:
                            exp += f"{' OR '.join(relevant_rules)}"
                        else:
                            exp += "Unknown reasoning path"
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
    Complete NEXUS-Transformer architecture
    """
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, num_constraints, symbol_space_size, symbol_names):
        # Initialize neural module
        self.neural_module = NeuralModule(vocab_size, embed_dim, num_heads, num_layers, num_constraints)
        
        # Initialize knowledge graph
        self.knowledge_graph = KnowledgeGraph()
        
        # Initialize symbolic module
        self.symbolic_module = SymbolicModule(self.knowledge_graph, symbol_space_size, embed_dim)
        
        # Initialize neural-symbolic interface
        self.interface = NeuralSymbolicInterface(self.neural_module, self.symbolic_module)
        
        # Initialize metacognitive controller
        self.metacognitive = MetacognitiveController()
        
        # Initialize explanation generator
        self.explanation_generator = ExplanationGenerator(self.knowledge_graph, symbol_names)
        
    def process(self, input_ids, attention_mask=None, risk_level='medium'):
        """
        Process input through the NEXUS-Transformer
        
        Args:
            input_ids: Token IDs of input text
            attention_mask: Attention mask for padding
            risk_level: Risk level of current task
            
        Returns:
            Dictionary with results and explanations
        """
        # Neural processing
        neural_output = self.neural_module(input_ids, attention_mask=attention_mask)
        
        # Extract neural representations
        neural_repr = neural_output['hidden_states']
        
        # Neural-to-symbolic translation
        active_symbols, symbol_similarities = self.interface.neural_to_symbolic(neural_repr)
        
        # Symbolic reasoning
        inferred_symbols = self.symbolic_module.reason(active_symbols)
        
        # Symbolic-to-neural translation
        symbolic_neural_repr = self.interface.symbolic_to_neural(inferred_symbols)
        
        # Compute confidences
        neural_confidence = torch.max(F.softmax(neural_output['logits'], dim=-1), dim=-1)[0].mean().item()
        symbolic_confidence = torch.mean(symbol_similarities.max(dim=1)[0]).item()
        
        # Metacognitive control
        strategy = self.metacognitive.decide_strategy(neural_confidence, symbolic_confidence, risk_level)
        
        # Generate final prediction
        if strategy['strategy'] == 'neural':
            final_logits = neural_output['logits']
        elif strategy['strategy'] == 'symbolic':
            # Project symbolic representation back to vocabulary space
            final_logits = self.neural_module.output_proj(symbolic_neural_repr)
        else:  # hybrid
            neural_weight = strategy['neural_weight']
            symbolic_weight = strategy['symbolic_weight']
            
            # Project symbolic representation back to vocabulary space
            symbolic_logits = self.neural_module.output_proj(symbolic_neural_repr)
            
            # Weighted combination
            final_logits = neural_weight * neural_output['logits'] + symbolic_weight * symbolic_logits
        
        # Generate explanation
        explanations = self.explanation_generator.generate_explanation(
            active_symbols, inferred_symbols, strategy, level='detailed'
        )
        
        return {
            'logits': final_logits,
            'active_symbols': active_symbols,
            'inferred_symbols': inferred_symbols,
            'strategy': strategy,
            'neural_confidence': neural_confidence,
            'symbolic_confidence': symbolic_confidence,
            'explanations': explanations
        }


# ===============================
# Example Usage
# ===============================

def create_medical_nexus_transformer():
    """Create a NEXUS-Transformer for medical diagnosis"""
    # Define vocabulary and symbols
    vocab_size = 1000  # Simplified vocabulary
    symbol_space_size = 20
    symbol_names = [
        'fever', 'high_fever', 'cough', 'severe_cough', 'fatigue', 
        'headache', 'shortness_of_breath', 'sore_throat', 
        'common_cold', 'flu', 'covid', 'pneumonia', 'allergies',
        'elderly', 'immunocompromised', 'child', 
        'need_hospitalization', 'need_ventilator', 'need_rest', 'need_testing'
    ]
    
    # Create NEXUS-Transformer
    model = NEXUSTransformer(
        vocab_size=vocab_size,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        num_constraints=5,
        symbol_space_size=symbol_space_size,
        symbol_names=symbol_names
    )
    
    # Build knowledge graph
    # Entity IDs correspond to their position in symbol_names list
    kg = model.knowledge_graph
    
    # Add entities (symptom, condition, patient type, treatment)
    for i, name in enumerate(symbol_names):
        kg.add_entity(i, name)
    
    # Add relations between symptoms and conditions
    kg.add_relation(0, 0, 9, 'symptom_of', 0.7)  # fever -> flu
    kg.add_relation(1, 1, 9, 'symptom_of', 0.9)  # high_fever -> flu
    kg.add_relation(2, 2, 8, 'symptom_of', 0.8)  # cough -> common_cold
    kg.add_relation(3, 2, 10, 'symptom_of', 0.7)  # cough -> covid
    kg.add_relation(4, 2, 9, 'symptom_of', 0.6)  # cough -> flu
    kg.add_relation(5, 3, 10, 'symptom_of', 0.8)  # severe_cough -> covid
    kg.add_relation(6, 3, 11, 'symptom_of', 0.7)  # severe_cough -> pneumonia
    kg.add_relation(7, 4, 8, 'symptom_of', 0.6)  # fatigue -> common_cold
    kg.add_relation(8, 4, 9, 'symptom_of', 0.7)  # fatigue -> flu
    kg.add_relation(9, 4, 10, 'symptom_of', 0.6)  # fatigue -> covid
    kg.add_relation(10, 5, 8, 'symptom_of', 0.5)  # headache -> common_cold
    kg.add_relation(11, 5, 9, 'symptom_of', 0.6)  # headache -> flu
    kg.add_relation(12, 6, 10, 'symptom_of', 0.8)  # shortness_of_breath -> covid
    kg.add_relation(13, 6, 11, 'symptom_of', 0.9)  # shortness_of_breath -> pneumonia
    kg.add_relation(14, 7, 8, 'symptom_of', 0.8)  # sore_throat -> common_cold
    
    # Add relations for risk factors
    kg.add_relation(15, 13, 16, 'risk_factor', 0.7)  # elderly -> need_hospitalization
    kg.add_relation(16, 14, 16, 'risk_factor', 0.8)  # immunocompromised -> need_hospitalization
    kg.add_relation(17, 13, 17, 'risk_factor', 0.6)  # elderly -> need_ventilator (if severe)
    kg.add_relation(18, 14, 17, 'risk_factor', 0.7)  # immunocompromised -> need_ventilator (if severe)
    
    # Add relations for treatments
    kg.add_relation(19, 8, 18, 'requires', 0.9)  # common_cold -> need_rest
    kg.add_relation(20, 9, 18, 'requires', 0.8)  # flu -> need_rest
    kg.add_relation(21, 9, 19, 'requires', 0.6)  # flu -> need_testing
    kg.add_relation(22, 10, 19, 'requires', 0.9)  # covid -> need_testing
    kg.add_relation(23, 10, 16, 'requires', 0.5)  # covid -> need_hospitalization
    kg.add_relation(24, 11, 16, 'requires', 0.8)  # pneumonia -> need_hospitalization
    kg.add_relation(25, 11, 17, 'requires', 0.7)  # pneumonia -> need_ventilator
    
    # Add logical rules
    # If high fever, cough, and fatigue, then flu
    kg.add_rule(0, [0, 2, 4], 9, 0.8)
    # If cough, shortness of breath, and fatigue, then covid
    kg.add_rule(1, [2, 6, 4], 10, 0.7)
    # If severe cough and shortness of breath, then pneumonia
    kg.add_rule(2, [3, 6], 11, 0.9)
    # If elderly or immunocompromised AND covid, then need hospitalization
    kg.add_rule(3, [13, 10], 16, 0.8)
    kg.add_rule(4, [14, 10], 16, 0.9)
    # If elderly or immunocompromised AND pneumonia, then need ventilator
    kg.add_rule(5, [13, 11], 17, 0.7)
    kg.add_rule(6, [14, 11], 17, 0.8)
    
    return model

def simulate_input(vocab_size, sequence):
    """Create toy input tensors for the model"""
    # Convert sequence of words to token IDs (simplified)
    # Make sure the sequence is within vocab range
    input_ids = torch.tensor([[min(i, vocab_size-1) for i in sequence]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    
    return input_ids, attention_mask

def demo_nexus_transformer():
    """Demonstrate the NEXUS-Transformer on medical diagnosis examples"""
    print("Initializing NEXUS-Transformer for medical diagnosis...")
    model = create_medical_nexus_transformer()
    
    # Example inputs representing different medical scenarios
    examples = [
        # Simple case: Common cold
        [1, 2, 7],  # Fever, cough, sore throat
        
        # Flu case
        [1, 2, 4, 5],  # Fever, cough, fatigue, headache
        
        # COVID case
        [1, 2, 4, 6],  # Fever, cough, fatigue, shortness of breath
        
        # Pneumonia case
        [1, 3, 6],  # Fever, severe cough, shortness of breath
        
        # High-risk patient (elderly with COVID symptoms)
        [1, 2, 6, 13],  # Fever, cough, shortness of breath, elderly
        
        # High-risk patient (immunocompromised with pneumonia symptoms)
        [1, 3, 6, 14],  # Fever, severe cough, shortness of breath, immunocompromised
    ]
    
    print("\nProcessing medical scenarios...\n")
    
    for i, sequence in enumerate(examples):
        print(f"Example {i+1}: {[model.explanation_generator.symbol_names[s] for s in sequence]}")
        
        # Create input tensors
        input_ids, attention_mask = simulate_input(model.neural_module.token_embedding.weight.size(0), sequence)
        
        # Process through NEXUS-Transformer
        with torch.no_grad():  # No need to track gradients for inference
            output = model.process(input_ids, attention_mask)
        
        # Print active and inferred symbols
        active_ids = torch.nonzero(output['active_symbols'][0]).squeeze(-1).tolist()
        inferred_ids = torch.nonzero(output['inferred_symbols'][0]).squeeze(-1).tolist()
        new_inferred = set(inferred_ids) - set(active_ids)
        
        print(f"Strategy: {output['strategy']['strategy']}")
        print(f"Neural confidence: {output['neural_confidence']:.4f}")
        print(f"Symbolic confidence: {output['symbolic_confidence']:.4f}")
        print(f"Active symbols: {[model.explanation_generator.symbol_names[i] for i in active_ids]}")
        print(f"Newly inferred: {[model.explanation_generator.symbol_names[i] for i in new_inferred]}")
        print("\nExplanation:")
        print(output['explanations'][0])
        print("-" * 80)
    
    return model

def demonstrate_self_improvement():
    """
    Demonstrate the self-improvement capability of NEXUS-Transformer
    
    This is a simplified demonstration of how the model could learn from experience
    """
    print("Demonstrating self-improvement capability...")
    model = create_medical_nexus_transformer()
    
    # Example of a misdiagnosed case
    input_sequence = [1, 2, 4, 6]  # Fever, cough, fatigue, shortness of breath
    input_ids, attention_mask = simulate_input(model.neural_module.token_embedding.weight.size(0), input_sequence)
    
    # Process through NEXUS-Transformer
    with torch.no_grad():
        output_before = model.process(input_ids, attention_mask)
    
    # Simulate feedback that the correct diagnosis was COVID (10)
    correct_diagnosis = 10  # COVID
    
    print("\nOriginal diagnosis:")
    active_ids = torch.nonzero(output_before['active_symbols'][0]).squeeze(-1).tolist()
    inferred_ids = torch.nonzero(output_before['inferred_symbols'][0]).squeeze(-1).tolist()
    print(f"Active symbols: {[model.explanation_generator.symbol_names[i] for i in active_ids]}")
    print(f"Inferred symbols: {[model.explanation_generator.symbol_names[i] for i in inferred_ids]}")
    
    # Add a new rule based on the feedback
    print("\nAdding new rule based on feedback...")
    model.knowledge_graph.add_rule(
        100,  # New rule ID
        [1, 2, 4, 6],  # Fever, cough, fatigue, shortness of breath
        correct_diagnosis,  # COVID
        0.9  # High confidence
    )
    
    # Process through NEXUS-Transformer again
    with torch.no_grad():
        output_after = model.process(input_ids, attention_mask)
    
    # Show improved diagnosis
    print("\nUpdated diagnosis after learning:")
    active_ids = torch.nonzero(output_after['active_symbols'][0]).squeeze(-1).tolist()
    inferred_ids = torch.nonzero(output_after['inferred_symbols'][0]).squeeze(-1).tolist()
    print(f"Active symbols: {[model.explanation_generator.symbol_names[i] for i in active_ids]}")
    print(f"Inferred symbols: {[model.explanation_generator.symbol_names[i] for i in inferred_ids]}")
    print("\nExplanation:")
    print(output_after['explanations'][0])

if __name__ == "__main__":
    print("NEXUS-Transformer: A Neural-Symbolic Architecture Demo")
    print("=" * 60)
    model = demo_nexus_transformer()
    print("\n")
    demonstrate_self_improvement()