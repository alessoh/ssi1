import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional, Tuple

class SimpleTransformerLayer(nn.Module):
    """
    Basic transformer layer with self-attention and feed-forward network
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, attention_mask=None):
        # Self-attention block with residual connection
        residual = x
        x = self.layer_norm1(x)
        attn_output, attn_weights = self.self_attn(x, x, x, attn_mask=attention_mask)
        x = residual + attn_output
        
        # Feed-forward block with residual connection
        residual = x
        x = self.layer_norm2(x)
        ff_output = self.ff_network(x)
        x = residual + ff_output
        
        return x, attn_weights

class SimpleKnowledgeGraph:
    """
    Simple knowledge graph implementation with basic reasoning
    """
    def __init__(self):
        # Initialize empty graph structure
        self.entities = {}  # entity_id -> name
        self.relations = []  # (source_id, relation_type, target_id, weight)
        self.rules = []  # (premise_ids, conclusion_id, confidence)
        
    def add_entity(self, entity_id, name):
        """Add an entity to the knowledge graph"""
        self.entities[entity_id] = name
        return self
        
    def add_relation(self, source_id, relation_type, target_id, weight=1.0):
        """Add a relation between two entities"""
        self.relations.append((source_id, relation_type, target_id, weight))
        return self
        
    def add_rule(self, premise_ids, conclusion_id, confidence=1.0):
        """Add a logical rule to the knowledge graph"""
        self.rules.append((premise_ids, conclusion_id, confidence))
        return self
    
    def reason(self, active_entities):
        """
        Apply simple forward reasoning to derive new knowledge
        
        Args:
            active_entities: Set of currently active entity IDs
            
        Returns:
            Set of inferred entity IDs and reasoning steps
        """
        inferred = set(active_entities)
        reasoning_steps = {}
        
        # Apply relation-based inference
        for source_id, relation_type, target_id, weight in self.relations:
            if source_id in inferred and target_id not in inferred:
                inferred.add(target_id)
                step = f"{self.entities[source_id]} --{relation_type}--> {self.entities[target_id]}"
                reasoning_steps[target_id] = step
        
        # Apply rule-based inference
        for premise_ids, conclusion_id, confidence in self.rules:
            if all(p_id in inferred for p_id in premise_ids) and conclusion_id not in inferred:
                inferred.add(conclusion_id)
                premises = [self.entities[p_id] for p_id in premise_ids]
                step = f"Rule: IF {' AND '.join(premises)} THEN {self.entities[conclusion_id]}"
                reasoning_steps[conclusion_id] = step
                
        return inferred, reasoning_steps

class NeuralSymbolicInterface:
    """
    Simple interface between neural and symbolic components
    """
    def __init__(self, input_dim, num_symbols):
        self.neural_to_symbol_matrix = torch.randn(num_symbols, input_dim)
        self.symbol_to_neural_matrix = torch.randn(input_dim, num_symbols)
        
    def neural_to_symbolic(self, neural_repr, threshold=0.5):
        """Convert neural representations to symbolic activations"""
        # Compute similarity between neural representations and symbol embeddings
        similarity = torch.matmul(neural_repr, self.neural_to_symbol_matrix.T)
        
        # Apply threshold to get binary activations
        activations = (similarity > threshold).float()
        
        return activations, similarity
    
def symbolic_to_neural(self, symbolic_activations):
    """Convert symbolic activations to neural representations"""
    # Compute weighted sum of symbol embeddings
    neural_repr = torch.matmul(symbolic_activations, self.symbol_to_neural_matrix.T)
    return neural_repr

class NeuralSymbolicInterface:
    """
    Simple interface between neural and symbolic components
    """
    def __init__(self, input_dim, num_symbols):
        self.neural_to_symbol_matrix = torch.randn(num_symbols, input_dim)
        self.symbol_to_neural_matrix = torch.randn(num_symbols, input_dim)
        
    def neural_to_symbolic(self, neural_repr, threshold=0.5):
        """Convert neural representations to symbolic activations"""
        # Compute similarity between neural representations and symbol embeddings
        similarity = torch.matmul(neural_repr, self.neural_to_symbol_matrix.T)
        
        # Apply threshold to get binary activations
        activations = (similarity > threshold).float()
        
        return activations, similarity
    
    # ADD THIS METHOD - it's missing in your code
    def symbolic_to_neural(self, symbolic_activations):
        """Convert symbolic activations to neural representations"""
        # Compute weighted sum of symbol embeddings
        neural_repr = torch.matmul(symbolic_activations, self.symbol_to_neural_matrix)
        
        return neural_repr
    
class NexusToyModel:
    """
    Simplified NEXUS architecture with basic neural and symbolic components
    """
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, num_symbols, symbol_names, dropout=0.1):
        # Neural component
        self.transformer_layers = [SimpleTransformerLayer(embed_dim, num_heads, ff_dim, dropout) 
                                  for _ in range(num_layers)]
        
        # Symbolic component
        self.knowledge_graph = SimpleKnowledgeGraph()
        
        # Neural-symbolic interface
        self.interface = NeuralSymbolicInterface(embed_dim, num_symbols)
        
        # Store symbol names for explanation
        self.symbol_names = symbol_names
        self.symbol_to_id = {name: i for i, name in enumerate(symbol_names)}
        
    def forward(self, neural_input):
        """
        Process input through neural and symbolic components
        
        Args:
            neural_input: Tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            Dictionary with neural and symbolic outputs
        """
        # Process through transformer layers
        x = neural_input
        all_attentions = []
        
        for layer in self.transformer_layers:
            x, attn_weights = layer(x)
            all_attentions.append(attn_weights)
            
        # Take final representation from first token (CLS)
        cls_representation = x[:, 0, :]
        
        # Convert to symbolic representation
        symbolic_activations, symbolic_scores = self.interface.neural_to_symbolic(cls_representation)
        
        # Apply reasoning
        batch_size = neural_input.size(0)
        all_inferred = []
        all_reasoning_steps = []
        
        for i in range(batch_size):
            # Get active symbols for this instance
            active_symbols = torch.nonzero(symbolic_activations[i]).squeeze(-1).tolist()
            if not isinstance(active_symbols, list):
                active_symbols = [active_symbols]
                
            # Apply reasoning
            inferred, reasoning_steps = self.knowledge_graph.reason(active_symbols)
            
            # Store results
            all_inferred.append(inferred)
            all_reasoning_steps.append(reasoning_steps)
            
        # Convert inferred symbols back to neural representation
        inferred_activations = torch.zeros_like(symbolic_activations)
        for i, inferred in enumerate(all_inferred):
            for symbol_id in inferred:
                if symbol_id < inferred_activations.size(1):
                    inferred_activations[i, symbol_id] = 1.0
                    
        enhanced_representation = self.interface.symbolic_to_neural(inferred_activations)
        
        # Combine neural and symbolic representations
        combined_representation = cls_representation + enhanced_representation
        
        return {
            'neural_representation': cls_representation,
            'symbolic_activations': symbolic_activations,
            'symbolic_scores': symbolic_scores,
            'inferred_symbols': all_inferred,
            'reasoning_steps': all_reasoning_steps,
            'enhanced_representation': enhanced_representation,
            'combined_representation': combined_representation
        }
    
    def explain(self, inferred_symbols, reasoning_steps):
        """Generate explanation for the inference process"""
        active_names = [self.symbol_names[i] for i in inferred_symbols if i < len(self.symbol_names)]
        
        explanation = [f"Identified concepts: {', '.join(active_names)}"]
        
        # Add reasoning steps
        if reasoning_steps:
            explanation.append("Reasoning steps:")
            for symbol_id, step in reasoning_steps.items():
                if symbol_id < len(self.symbol_names):
                    symbol_name = self.symbol_names[symbol_id]
                    explanation.append(f"  - {symbol_name}: {step}")
        
        return "\n".join(explanation)

# Example usage for a simple medical diagnosis model
def create_toy_medical_model():
    # Model parameters
    embed_dim = 64
    num_heads = 4
    ff_dim = 128
    num_layers = 2
    
    # Define medical symbols
    symbols = [
        'fever', 'cough', 'fatigue', 'shortness_of_breath',
        'common_cold', 'influenza', 'covid19', 'pneumonia',
        'rest', 'medication', 'hospitalization'
    ]
    num_symbols = len(symbols)
    
    # Create model
    model = NexusToyModel(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        num_symbols=num_symbols,
        symbol_names=symbols
    )
    
    # Initialize knowledge graph with medical knowledge
    kg = model.knowledge_graph
    
    # Add entities
    for i, symbol in enumerate(symbols):
        kg.add_entity(i, symbol)
    
    # Add relations
    kg.add_relation(0, "symptom_of", 5)  # fever -> influenza
    kg.add_relation(0, "symptom_of", 6)  # fever -> covid19
    kg.add_relation(1, "symptom_of", 4)  # cough -> common_cold
    kg.add_relation(1, "symptom_of", 5)  # cough -> influenza
    kg.add_relation(1, "symptom_of", 6)  # cough -> covid19
    kg.add_relation(1, "symptom_of", 7)  # cough -> pneumonia
    kg.add_relation(2, "symptom_of", 5)  # fatigue -> influenza
    kg.add_relation(2, "symptom_of", 6)  # fatigue -> covid19
    kg.add_relation(3, "symptom_of", 6)  # shortness_of_breath -> covid19
    kg.add_relation(3, "symptom_of", 7)  # shortness_of_breath -> pneumonia
    
    # Add logical rules
    kg.add_rule([0, 1, 2], 5)  # fever + cough + fatigue -> influenza
    kg.add_rule([0, 1, 3], 6)  # fever + cough + shortness_of_breath -> covid19
    kg.add_rule([4], 8)        # common_cold -> rest
    kg.add_rule([5], 9)        # influenza -> medication
    kg.add_rule([6], 9)        # covid19 -> medication
    kg.add_rule([7], 10)       # pneumonia -> hospitalization
    kg.add_rule([6, 3], 10)    # covid19 + shortness_of_breath -> hospitalization
    
    return model

# Example of using the toy model
def test_toy_model():
    # Create model
    model = create_toy_medical_model()
    
    # Create a dummy input (batch_size=1, seq_len=5, embed_dim=64)
    batch_size = 1
    seq_len = 5
    embed_dim = 64
    dummy_input = torch.randn(batch_size, seq_len, embed_dim)
    
    # Process input
    output = model.forward(dummy_input)
    
    # Create explanation
    explanation = model.explain(
        output['inferred_symbols'][0],
        output['reasoning_steps'][0]
    )
    
    print(explanation)
    
    return output, explanation

if __name__ == "__main__":
    test_toy_model()