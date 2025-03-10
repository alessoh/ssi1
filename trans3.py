import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# ===============================
# Neural Components
# ===============================

class NeuroSymbolicAttention(nn.Module):
    """
    Enhanced attention mechanism with deep integration of symbolic knowledge
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Standard projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Knowledge integration projections
        self.kg_q_proj = nn.Linear(embed_dim, embed_dim)
        self.kg_k_proj = nn.Linear(embed_dim, embed_dim)
        self.kg_v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Relation-specific transformations
        self.relation_proj = nn.Linear(embed_dim, num_heads)
        
        # Gating mechanism
        self.knowledge_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, kg_embeddings=None, relation_weights=None, attention_mask=None):
        batch_size = query.shape[0]
        
        # Standard attention mechanism
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate standard attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply knowledge graph influence if provided
        if kg_embeddings is not None:
            # Project knowledge graph embeddings
            kg_q = self.kg_q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            kg_k = self.kg_k_proj(kg_embeddings).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            kg_v = self.kg_v_proj(kg_embeddings).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Calculate knowledge-aware attention scores
            kg_scores = torch.matmul(kg_q, kg_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Apply relation-specific weights if provided
            if relation_weights is not None:
                relation_attention = self.relation_proj(relation_weights).permute(0, 2, 1).unsqueeze(2)
                kg_scores = kg_scores + relation_attention
            
            # Determine the influence of knowledge using the gating mechanism
            gate_input = torch.cat([query, kg_embeddings[:, :query.size(1), :]], dim=-1)
            gate = self.knowledge_gate(gate_input).unsqueeze(1).unsqueeze(-1)
            
            # Combine standard and knowledge-aware scores
            scores = (1 - gate) * scores + gate * kg_scores
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Calculate output using attention weights
        attn_output = torch.matmul(attn_weights, v)
        
        # If knowledge embeddings are provided, also attend to them
        if kg_embeddings is not None:
            kg_output = torch.matmul(attn_weights, kg_v)
            # Combine with standard output using the same gate
            attn_output = (1 - gate) * attn_output + gate * kg_output
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class NeuroSymbolicTransformerLayer(nn.Module):
    """
    Transformer layer enhanced with neural-symbolic integration capabilities
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = NeuroSymbolicAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network with logical activation
        self.ff_network = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),  # Approximates logical functions better than ReLU
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Symbolic constraint mechanism
        self.constraint_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, kg_embeddings=None, relation_weights=None, attention_mask=None, constraints=None):
        # Self-attention block with residual connection
        residual = x
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.attention(
            x_norm, x_norm, x_norm, kg_embeddings, relation_weights, attention_mask
        )
        x = residual + attn_output
        
        # Feed-forward block with residual connection
        residual = x
        x_norm = self.norm2(x)
        ff_output = self.ff_network(x_norm)
        
        # Apply symbolic constraints if provided
        if constraints is not None:
            constraint_gate = self.constraint_gate(constraints)
            ff_output = ff_output * constraint_gate + constraints * (1 - constraint_gate)
            
        x = residual + ff_output
        
        return x, attn_weights


# ===============================
# Symbolic Components
# ===============================

class KnowledgeGraph:
    """
    Enhanced knowledge graph with reasoning capabilities
    """
    def __init__(self):
        # Initialize data structures
        self.entities = {}  # entity_id -> {name, embedding, attributes}
        self.relations = {}  # relation_id -> {source, target, type, weight}
        self.rules = {}  # rule_id -> {premise, conclusion, confidence}
        
    def add_entity(self, entity_id, name, embedding=None, attributes=None):
        """Add an entity to the knowledge graph"""
        self.entities[entity_id] = {
            'name': name,
            'embedding': embedding,
            'attributes': attributes or {}
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
        entity = self.entities.get(entity_id)
        if entity and entity.get('embedding') is not None:
            return entity['embedding']
        return None
    
    def get_related_entities(self, entity_id, relation_type=None):
        """Get entities related to a given entity"""
        related = []
        for rel_id, rel in self.relations.items():
            if rel['source'] == entity_id and (relation_type is None or rel['type'] == relation_type):
                related.append((rel['target'], rel['type'], rel['weight']))
            if rel['target'] == entity_id and (relation_type is None or rel['type'] == relation_type):
                related.append((rel['source'], rel['type'], rel['weight']))
        return related
    
    def reason(self, active_entities, max_hops=3):
        """
        Apply multi-hop reasoning to infer new knowledge
        
        Args:
            active_entities: Set of currently active entity IDs
            max_hops: Maximum number of reasoning hops
            
        Returns:
            Set of inferred entity IDs, reasoning paths
        """
        inferred = set(active_entities)
        all_paths = {}  # entity_id -> reasoning path
        
        # Initialize paths for active entities
        for entity_id in active_entities:
            all_paths[entity_id] = [("Given", entity_id)]
        
        # Perform multi-hop reasoning
        for hop in range(max_hops):
            new_inferred = set()
            new_paths = {}
            
            # Apply relation-based inference
            for entity_id in inferred:
                related = self.get_related_entities(entity_id)
                for target_id, relation_type, weight in related:
                    if target_id not in inferred and target_id not in new_inferred:
                        new_inferred.add(target_id)
                        new_paths[target_id] = all_paths.get(entity_id, []) + [("Relation", entity_id, relation_type, target_id)]
            
            # Apply rule-based inference
            for rule_id, rule in self.rules.items():
                premise_entities = rule['premise']
                conclusion_entity = rule['conclusion']
                
                # Check if all premise entities are active
                if all(entity in inferred for entity in premise_entities):
                    if conclusion_entity not in inferred and conclusion_entity not in new_inferred:
                        new_inferred.add(conclusion_entity)
                        new_paths[conclusion_entity] = [("Rule", rule_id, premise_entities, conclusion_entity)]
            
            # No new inferences, stop early
            if not new_inferred:
                break
                
            # Update inferred entities and paths
            inferred.update(new_inferred)
            all_paths.update(new_paths)
        
        return inferred, all_paths


class SymbolGrounder(nn.Module):
    """
    Neural network that maps between neural representations and symbolic concepts
    """
    def __init__(self, embed_dim, num_symbols, hidden_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_symbols = num_symbols
        hidden_dim = hidden_dim or embed_dim // 2
        
        # Neural to symbolic mapping
        self.n2s_network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_symbols)
        )
        
        # Symbolic to neural mapping
        self.s2n_embed = nn.Embedding(num_symbols, embed_dim)
        self.s2n_attention = nn.MultiheadAttention(embed_dim, 4, batch_first=True)
        
    def neural_to_symbolic(self, neural_repr, threshold=0.5, temperature=1.0):
        """Map neural representations to symbolic concepts"""
        # Apply neural network to get logits
        symbol_logits = self.n2s_network(neural_repr)
        
        # Apply temperature scaling
        scaled_logits = symbol_logits / temperature
        
        # Get probabilities and binary activations
        symbol_probs = torch.sigmoid(scaled_logits)
        symbol_activations = (symbol_probs > threshold).float()
        
        return {
            'logits': symbol_logits,
            'probabilities': symbol_probs,
            'activations': symbol_activations
        }
    
    def symbolic_to_neural(self, symbol_activations, neural_context=None):
        """Map symbolic concepts to neural representations"""
        batch_size = symbol_activations.shape[0]
        
        # Get embeddings for active symbols
        # For each batch item, we need to get embeddings for active symbols
        all_embeddings = []
        
        for i in range(batch_size):
            # Get indices of active symbols
            active_indices = torch.nonzero(symbol_activations[i]).squeeze(-1)
            
            if len(active_indices.shape) == 0:  # Handle case of single active index
                active_indices = active_indices.unsqueeze(0)
                
            if len(active_indices) == 0:  # Handle case of no active symbols
                # Use average of all symbol embeddings as fallback
                embeddings = self.s2n_embed.weight.mean(dim=0, keepdim=True)
            else:
                # Get embeddings for active symbols
                embeddings = self.s2n_embed(active_indices)
            
            all_embeddings.append(embeddings)
        
        # If neural context is provided, use attention to combine embeddings
        if neural_context is not None:
            neural_repr = []
            
            for i, embeddings in enumerate(all_embeddings):
                # Use the neural context to attend to symbol embeddings
                context = neural_context[i].unsqueeze(0)  # [1, seq_len, embed_dim]
                
                if embeddings.size(0) > 0:  # Check if there are embeddings
                    # Apply attention
                    attn_output, _ = self.s2n_attention(
                        context, 
                        embeddings.unsqueeze(0),  # [1, num_active, embed_dim]
                        embeddings.unsqueeze(0)
                    )
                    neural_repr.append(attn_output.squeeze(0))  # [seq_len, embed_dim]
                else:
                    # Fallback: just use the context
                    neural_repr.append(context.squeeze(0))
            
            neural_repr = torch.stack(neural_repr)  # [batch_size, seq_len, embed_dim]
        else:
            # Without context, just average the embeddings
            neural_repr = torch.stack([emb.mean(dim=0, keepdim=True) for emb in all_embeddings])
            neural_repr = neural_repr.squeeze(1)  # [batch_size, embed_dim]
            
        return neural_repr


# ===============================
# Integration Components
# ===============================

class MetacognitiveController:
    """
    Enhanced controller that determines the processing strategy
    """
    def __init__(self, neural_threshold=0.7, symbolic_threshold=0.8):
        self.neural_threshold = neural_threshold
        self.symbolic_threshold = symbolic_threshold
        
    def compute_confidence(self, neural_output, symbolic_output):
        """Compute confidence metrics for neural and symbolic outputs"""
        # Neural confidence: max probability of the prediction
        if isinstance(neural_output, dict) and 'probabilities' in neural_output:
            neural_conf = neural_output['probabilities'].max(dim=-1)[0].mean().item()
        elif isinstance(neural_output, torch.Tensor):
            # Assume logits
            neural_conf = F.softmax(neural_output, dim=-1).max(dim=-1)[0].mean().item()
        else:
            neural_conf = 0.5  # Default
            
        # Symbolic confidence
        if isinstance(symbolic_output, dict) and 'confidence' in symbolic_output:
            symbolic_conf = symbolic_output['confidence']
        elif isinstance(symbolic_output, tuple) and len(symbolic_output) > 1:
            symbolic_conf = symbolic_output[1]  # Assume (output, confidence) tuple
        else:
            symbolic_conf = 0.5  # Default
            
        return neural_conf, symbolic_conf
    
    def decide_strategy(self, neural_conf, symbolic_conf, task_type='general', risk_level='medium'):
        """
        Decide which processing strategy to use
        
        Args:
            neural_conf: Confidence of neural prediction
            symbolic_conf: Confidence of symbolic prediction
            task_type: Type of task ('reasoning', 'pattern', 'general')
            risk_level: Risk level of the current task ('low', 'medium', 'high')
            
        Returns:
            Strategy dictionary
        """
        # Adjust thresholds based on task type and risk level
        neural_threshold = self.neural_threshold
        symbolic_threshold = self.symbolic_threshold
        
        # Task type adjustments
        if task_type == 'reasoning':
            # Favor symbolic for reasoning tasks
            neural_threshold += 0.1
            symbolic_threshold -= 0.1
        elif task_type == 'pattern':
            # Favor neural for pattern recognition tasks
            neural_threshold -= 0.1
            symbolic_threshold += 0.1
            
        # Risk level adjustments
        if risk_level == 'high':
            # Be more cautious with high-risk tasks
            neural_threshold += 0.1
            symbolic_threshold -= 0.1
        elif risk_level == 'low':
            # Be more lenient with low-risk tasks
            neural_threshold -= 0.1
            symbolic_threshold += 0.1
            
        # Determine strategy
        if neural_conf >= neural_threshold and symbolic_conf < symbolic_threshold:
            return {
                'strategy': 'neural',
                'neural_weight': 1.0,
                'symbolic_weight': 0.0,
                'explanation': f'Using neural prediction due to high neural confidence ({neural_conf:.2f})'
            }
        elif symbolic_conf >= symbolic_threshold and neural_conf < neural_threshold:
            return {
                'strategy': 'symbolic',
                'neural_weight': 0.0,
                'symbolic_weight': 1.0,
                'explanation': f'Using symbolic reasoning due to high symbolic confidence ({symbolic_conf:.2f})'
            }
        else:
            # Use weighted combination
            total_conf = neural_conf + symbolic_conf
            neural_weight = neural_conf / total_conf if total_conf > 0 else 0.5
            symbolic_weight = symbolic_conf / total_conf if total_conf > 0 else 0.5
            
            return {
                'strategy': 'hybrid',
                'neural_weight': neural_weight,
                'symbolic_weight': symbolic_weight,
                'explanation': f'Using weighted combination (neural: {neural_weight:.2f}, symbolic: {symbolic_weight:.2f})'
            }


class ExplanationGenerator:
    """
    Enhanced explanation generator with multiple levels of detail
    """
    def __init__(self, knowledge_graph, symbol_names):
        self.kg = knowledge_graph
        self.symbol_names = symbol_names
        
    def get_entity_name(self, entity_id):
        """Get the name of an entity"""
        if isinstance(entity_id, str):
            return entity_id
        
        if entity_id < len(self.symbol_names):
            return self.symbol_names[entity_id]
        
        entity = self.kg.entities.get(entity_id)
        if entity:
            return entity['name']
        
        return f"Entity_{entity_id}"
    
    def format_path(self, path_step):
        """Format a single reasoning path step"""
        step_type = path_step[0]
        
        if step_type == "Given":
            entity_id = path_step[1]
            return f"Given: {self.get_entity_name(entity_id)}"
        
        elif step_type == "Relation":
            source_id, relation_type, target_id = path_step[1:4]
            return f"{self.get_entity_name(source_id)} --[{relation_type}]--> {self.get_entity_name(target_id)}"
        
        elif step_type == "Rule":
            rule_id, premises, conclusion = path_step[1:4]
            premise_str = ", ".join(self.get_entity_name(p) for p in premises)
            return f"Rule {rule_id}: IF {premise_str} THEN {self.get_entity_name(conclusion)}"
        
        return str(path_step)
    
    def generate_explanation(self, 
                             active_symbols, 
                             inferred_symbols, 
                             reasoning_paths, 
                             strategy,
                             level='simple'):
        """
        Generate explanation for the current prediction
        
        Args:
            active_symbols: Tensor or list of active symbol IDs
            inferred_symbols: Tensor or list of inferred symbol IDs
            reasoning_paths: Dictionary mapping entity IDs to reasoning paths
            strategy: Strategy dictionary from metacognitive controller
            level: Detail level ('simple', 'detailed', 'technical')
            
        Returns:
            Explanation string
        """
        # Convert tensors to lists if needed
        if isinstance(active_symbols, torch.Tensor):
            active_ids = torch.nonzero(active_symbols).squeeze(-1).tolist()
            if not isinstance(active_ids, list):
                active_ids = [active_ids]
        else:
            active_ids = active_symbols
            
        if isinstance(inferred_symbols, torch.Tensor):
            inferred_ids = torch.nonzero(inferred_symbols).squeeze(-1).tolist()
            if not isinstance(inferred_ids, list):
                inferred_ids = [inferred_ids]
        else:
            inferred_ids = inferred_symbols
        
        # Get names
        active_names = [self.get_entity_name(idx) for idx in active_ids]
        inferred_names = [self.get_entity_name(idx) for idx in inferred_ids if idx not in active_ids]
        
        # Generate explanation based on level
        if level == 'simple':
            explanation = [
                f"Strategy: {strategy['strategy']}",
                f"Identified concepts: {', '.join(active_names) if active_names else 'None'}"
            ]
            
            if inferred_names:
                explanation.append(f"Inferred concepts: {', '.join(inferred_names)}")
            else:
                explanation.append("No additional concepts inferred.")
                
        elif level == 'detailed':
            explanation = [
                f"Strategy: {strategy['strategy']} (Neural weight: {strategy.get('neural_weight', 0):.2f}, "
                f"Symbolic weight: {strategy.get('symbolic_weight', 0):.2f})",
                f"Identified concepts: {', '.join(active_names) if active_names else 'None'}"
            ]
            
            if inferred_names:
                explanation.append(f"Inferred concepts: {', '.join(inferred_names)}")
                explanation.append("Reasoning paths:")
                
                for entity_id in inferred_ids:
                    if entity_id in active_ids:
                        continue  # Skip already active entities
                        
                    if entity_id in reasoning_paths:
                        path = reasoning_paths[entity_id]
                        path_str = " -> ".join(self.format_path(step) for step in path)
                        explanation.append(f"  - {self.get_entity_name(entity_id)}: {path_str}")
                    else:
                        explanation.append(f"  - {self.get_entity_name(entity_id)}: Unknown reasoning path")
            else:
                explanation.append("No additional concepts inferred.")
                
        else:  # technical
            explanation = [
                f"Strategy: {strategy['strategy']} with weights ["
                f"{strategy.get('neural_weight', 0):.4f}, {strategy.get('symbolic_weight', 0):.4f}]",
                f"Active symbols (IDs): {active_ids}",
                f"Inferred symbols (IDs): {[e for e in inferred_ids if e not in active_ids]}",
                f"Active symbol names: {active_names}",
                f"Inferred symbol names: {inferred_names}"
            ]
            
            # Add detailed reasoning paths
            if inferred_names:
                explanation.append("Detailed reasoning paths:")
                for entity_id in inferred_ids:
                    if entity_id in active_ids:
                        continue  # Skip already active entities
                        
                    if entity_id in reasoning_paths:
                        path = reasoning_paths[entity_id]
                        explanation.append(f"  - {self.get_entity_name(entity_id)}:")
                        for i, step in enumerate(path):
                            explanation.append(f"    {i+1}. {self.format_path(step)}")
                    else:
                        explanation.append(f"  - {self.get_entity_name(entity_id)}: No reasoning path available")
        
        return "\n".join(explanation)


# ===============================
# Full NEXUS Transformer
# ===============================

class NEXUSTransformer(nn.Module):
    """
    Complete neural-symbolic transformer with integrated reasoning capabilities
    """
    def __init__(self, 
                vocab_size, 
                embed_dim, 
                num_heads, 
                ff_dim, 
                num_layers, 
                num_symbols, 
                symbol_names,
                dropout=0.1,
                max_seq_len=512):
        super().__init__()
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_encoding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            NeuroSymbolicTransformerLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Symbol grounding
        self.symbol_grounder = SymbolGrounder(embed_dim, num_symbols)
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # Knowledge graph
        self.knowledge_graph = KnowledgeGraph()
        
        # Metacognitive controller
        self.metacognitive = MetacognitiveController()
        
        # Explanation generator
        self.explanation_generator = ExplanationGenerator(self.knowledge_graph, symbol_names)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the weights"""
        # Token embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        
        # Position encoding
        nn.init.normal_(self.position_encoding, mean=0, std=0.02)
        
    def _prepare_inputs(self, input_ids, attention_mask=None):
        """Prepare the inputs for the model"""
        # Get sequence length
        seq_length = input_ids.size(1)
        
        # Get token embeddings
        embeddings = self.token_embedding(input_ids)
        
        # Add position encodings
        embeddings = embeddings + self.position_encoding[:, :seq_length, :]
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        return embeddings, attention_mask
    
    def forward(self, input_ids, attention_mask=None, task_type='general', risk_level='medium'):
        """
        Process input through the NEXUS Transformer
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            task_type: Type of task ('reasoning', 'pattern', 'general')
            risk_level: Risk level ('low', 'medium', 'high')
            
        Returns:
            Dictionary with results and explanations
        """
        # Prepare inputs
        embeddings, attention_mask = self._prepare_inputs(input_ids, attention_mask)
        
        # Process through transformer layers
        hidden_states = embeddings
        all_attention_weights = []
        
        for layer in self.transformer_layers:
            hidden_states, attn_weights = layer(
                hidden_states, 
                attention_mask=attention_mask
            )
            all_attention_weights.append(attn_weights)
        
        # Map to symbolic space
        symbolic_output = self.symbol_grounder.neural_to_symbolic(hidden_states[:, 0, :])  # Use CLS token
        active_symbols = symbolic_output['activations']
        
        # Perform symbolic reasoning
        batch_size = input_ids.size(0)
        all_inferred_symbols = []
        all_reasoning_paths = []
        
        for i in range(batch_size):
            # Get active symbols for this batch item
            batch_active = torch.nonzero(active_symbols[i]).squeeze(-1).tolist()
            if not isinstance(batch_active, list):
                batch_active = [batch_active]
                
            # Perform reasoning
            inferred, paths = self.knowledge_graph.reason(batch_active)
            
            # Convert to tensor
            inferred_tensor = torch.zeros_like(active_symbols[i])
            for idx in inferred:
                if idx < inferred_tensor.size(0):
                    inferred_tensor[idx] = 1.0
            
            all_inferred_symbols.append(inferred_tensor)
            all_reasoning_paths.append(paths)
        
        # Stack inferred symbols
        inferred_symbols = torch.stack(all_inferred_symbols)
        
        # Map inferred symbols back to neural space
        symbolic_neural = self.symbol_grounder.symbolic_to_neural(inferred_symbols, hidden_states)
        
        # Project to vocabulary (neural path)
        neural_logits = self.output_projection(hidden_states)
        
        # Project to vocabulary (symbolic path)
        symbolic_logits = self.output_projection(symbolic_neural)
        
        # Compute confidences
        neural_probs = F.softmax(neural_logits, dim=-1)
        neural_conf = neural_probs.max(dim=-1)[0].mean().item()
        
        # Compute symbolic confidence based on reasoning paths
        symbolic_conf = 0.0
        for i, paths in enumerate(all_reasoning_paths):
            if paths:
                # Higher confidence with more reasoning steps
                path_lengths = [len(p) for p in paths.values()]
                avg_path_length = sum(path_lengths) / len(path_lengths) if path_lengths else 0
                # Normalize: longer paths -> higher confidence (up to a point)
                normalized_length = min(1.0, avg_path_length / 5.0)
                symbolic_conf += normalized_length
        
        symbolic_conf = symbolic_conf / batch_size if batch_size > 0 else 0.0
        
        # Metacognitive control
        strategy = self.metacognitive.decide_strategy(
            neural_conf, 
            symbolic_conf,
            task_type=task_type,
            risk_level=risk_level
        )
        
        # Select final output based on strategy
        if strategy['strategy'] == 'neural':
            final_logits = neural_logits
        elif strategy['strategy'] == 'symbolic':
            final_logits = symbolic_logits
        else:  # hybrid
            neural_weight = strategy['neural_weight']
            symbolic_weight = strategy['symbolic_weight']
            final_logits = neural_weight * neural_logits + symbolic_weight * symbolic_logits
        
        # Generate explanations
        explanations = []
        for i in range(batch_size):
            explanation = self.explanation_generator.generate_explanation(
                active_symbols[i],
                inferred_symbols[i],
                all_reasoning_paths[i],
                strategy,
                level='detailed'
            )
            explanations.append(explanation)
        
        return {
            'logits': final_logits,
            'neural_logits': neural_logits,
            'symbolic_logits': symbolic_logits,
            'hidden_states': hidden_states,
            'active_symbols': active_symbols,
            'inferred_symbols': inferred_symbols,
            'attention_weights': all_attention_weights,
            'neural_confidence': neural_conf,
            'symbolic_confidence': symbolic_conf,
            'strategy': strategy,
            'explanations': explanations
        }
    
    def predict(self, input_ids, attention_mask=None, task_type='general', risk_level='medium'):
        """
        Generate predictions from the model
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            task_type: Type of task
            risk_level: Risk level
            
        Returns:
            Predictions and explanations
        """
        # Process through model
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, task_type, risk_level)
        
        # Get predictions
        logits = outputs['logits']
        predictions = torch.argmax(logits, dim=-1)
        
        return {
            'predictions': predictions,
            'active_symbols': outputs['active_symbols'],
            'inferred_symbols': outputs['inferred_symbols'],
            'strategy': outputs['strategy'],
            'explanations': outputs['explanations']
        }
    
    def update_knowledge_graph(self, new_entity=None, new_relation=None, new_rule=None):
        """
        Update the knowledge graph with new information
        
        Args:
            new_entity: Tuple of (entity_id, name, embedding, attributes)
            new_relation: Tuple of (relation_id, source_id, target_id, relation_type, weight)
            new_rule: Tuple of (rule_id, premise, conclusion, confidence)
            
        Returns:
            Success flag
        """
        if new_entity:
            entity_id, name, embedding, attributes = new_entity
            self.knowledge_graph.add_entity(entity_id, name, embedding, attributes)
            
        if new_relation:
            relation_id, source_id, target_id, relation_type, weight = new_relation
            self.knowledge_graph.add_relation(relation_id, source_id, target_id, relation_type, weight)
            
        if new_rule:
            rule_id, premise, conclusion, confidence = new_rule
            self.knowledge_graph.add_rule(rule_id, premise, conclusion, confidence)
            
        return True
    
    def train_symbol_grounding(self, inputs, symbols, optimizer, epochs=5):
        """
        Train the symbol grounding component
        
        Args:
            inputs: Input tensors [batch_size, seq_len, embed_dim]
            symbols: Target symbols [batch_size, num_symbols]
            optimizer: Optimizer
            epochs: Number of training epochs
            
        Returns:
            Training losses
        """
        losses = []
        
        for epoch in range(epochs):
            # Get hidden representation
            with torch.no_grad():
                embeddings, _ = self._prepare_inputs(inputs)
                hidden_states = embeddings
                
                for layer in self.transformer_layers:
                    hidden_states, _ = layer(hidden_states)
            
            # Train symbol grounding
            cls_embeddings = hidden_states[:, 0, :]  # Use CLS token
            
            # Forward pass
            symbolic_output = self.symbol_grounder.neural_to_symbolic(cls_embeddings)
            logits = symbolic_output['logits']
            
            # Compute loss
            loss = F.binary_cross_entropy_with_logits(logits, symbols)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
        return losses

# ===============================
# Demonstration Functions
# ===============================

def create_medical_nexus_model(vocab_size=10000, max_seq_len=128):
    """
    Create a NEXUS Transformer model for medical diagnosis
    """
    # Model parameters
    embed_dim = 256
    num_heads = 8
    ff_dim = 1024
    num_layers = 4
    num_symbols = 50
    dropout = 0.1
    
    # Symbol names for medical domain
    symbol_names = [
        # Symptoms
        'fever', 'high_fever', 'cough', 'severe_cough', 'fatigue', 'severe_fatigue',
        'headache', 'severe_headache', 'shortness_of_breath', 'severe_shortness_of_breath',
        'sore_throat', 'runny_nose', 'body_aches', 'loss_of_taste', 'loss_of_smell',
        
        # Conditions
        'common_cold', 'influenza', 'covid19', 'pneumonia', 'bronchitis',
        'bacterial_infection', 'viral_infection', 'asthma_exacerbation',
        
        # Risk factors
        'elderly', 'immunocompromised', 'diabetes', 'heart_disease', 'lung_disease',
        'obesity', 'smoking_history', 'cancer',
        
        # Severity levels
        'mild_condition', 'moderate_condition', 'severe_condition', 'critical_condition',
        
        # Actions
        'needs_rest', 'needs_fluids', 'needs_medication', 'needs_antibiotics',
        'needs_antivirals', 'needs_testing', 'needs_monitoring', 'needs_hospitalization',
        'needs_ventilation', 'needs_icu', 'needs_emergency_care'
    ]
    
    # Create model
    model = NEXUSTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        num_symbols=num_symbols,
        symbol_names=symbol_names,
        dropout=dropout,
        max_seq_len=max_seq_len
    )
    
    # Initialize knowledge graph
    kg = model.knowledge_graph
    
    # Add entities (each symbol becomes an entity)
    for i, name in enumerate(symbol_names):
        kg.add_entity(i, name)
    
    # Add relationships between symptoms and conditions
    # Common cold relationships
    kg.add_relation(0, symbol_names.index('runny_nose'), symbol_names.index('common_cold'), 'symptom_of', 0.8)
    kg.add_relation(1, symbol_names.index('sore_throat'), symbol_names.index('common_cold'), 'symptom_of', 0.7)
    kg.add_relation(2, symbol_names.index('cough'), symbol_names.index('common_cold'), 'symptom_of', 0.6)
    kg.add_relation(3, symbol_names.index('headache'), symbol_names.index('common_cold'), 'symptom_of', 0.5)
    
    # Flu relationships
    kg.add_relation(4, symbol_names.index('fever'), symbol_names.index('influenza'), 'symptom_of', 0.8)
    kg.add_relation(5, symbol_names.index('body_aches'), symbol_names.index('influenza'), 'symptom_of', 0.8)
    kg.add_relation(6, symbol_names.index('fatigue'), symbol_names.index('influenza'), 'symptom_of', 0.7)
    kg.add_relation(7, symbol_names.index('cough'), symbol_names.index('influenza'), 'symptom_of', 0.6)
    
    # COVID-19 relationships
    kg.add_relation(8, symbol_names.index('fever'), symbol_names.index('covid19'), 'symptom_of', 0.7)
    kg.add_relation(9, symbol_names.index('cough'), symbol_names.index('covid19'), 'symptom_of', 0.7)
    kg.add_relation(10, symbol_names.index('fatigue'), symbol_names.index('covid19'), 'symptom_of', 0.6)
    kg.add_relation(11, symbol_names.index('shortness_of_breath'), symbol_names.index('covid19'), 'symptom_of', 0.6)
    kg.add_relation(12, symbol_names.index('loss_of_taste'), symbol_names.index('covid19'), 'symptom_of', 0.9)
    kg.add_relation(13, symbol_names.index('loss_of_smell'), symbol_names.index('covid19'), 'symptom_of', 0.9)
    
    # Pneumonia relationships
    kg.add_relation(14, symbol_names.index('fever'), symbol_names.index('pneumonia'), 'symptom_of', 0.7)
    kg.add_relation(15, symbol_names.index('cough'), symbol_names.index('pneumonia'), 'symptom_of', 0.8)
    kg.add_relation(16, symbol_names.index('shortness_of_breath'), symbol_names.index('pneumonia'), 'symptom_of', 0.9)
    kg.add_relation(17, symbol_names.index('fatigue'), symbol_names.index('pneumonia'), 'symptom_of', 0.6)
    
    # Severity relationships
    kg.add_relation(18, symbol_names.index('high_fever'), symbol_names.index('severe_condition'), 'indicates', 0.7)
    kg.add_relation(19, symbol_names.index('severe_shortness_of_breath'), symbol_names.index('severe_condition'), 'indicates', 0.9)
    kg.add_relation(20, symbol_names.index('severe_fatigue'), symbol_names.index('severe_condition'), 'indicates', 0.6)
    
    # Risk factor relationships
    kg.add_relation(21, symbol_names.index('elderly'), symbol_names.index('severe_condition'), 'increases_risk', 0.7)
    kg.add_relation(22, symbol_names.index('immunocompromised'), symbol_names.index('severe_condition'), 'increases_risk', 0.8)
    kg.add_relation(23, symbol_names.index('diabetes'), symbol_names.index('severe_condition'), 'increases_risk', 0.6)
    kg.add_relation(24, symbol_names.index('heart_disease'), symbol_names.index('severe_condition'), 'increases_risk', 0.7)
    kg.add_relation(25, symbol_names.index('lung_disease'), symbol_names.index('severe_condition'), 'increases_risk', 0.8)
    
    # Treatment relationships
    kg.add_relation(26, symbol_names.index('common_cold'), symbol_names.index('needs_rest'), 'requires', 0.9)
    kg.add_relation(27, symbol_names.index('common_cold'), symbol_names.index('needs_fluids'), 'requires', 0.9)
    
    kg.add_relation(28, symbol_names.index('influenza'), symbol_names.index('needs_rest'), 'requires', 0.9)
    kg.add_relation(29, symbol_names.index('influenza'), symbol_names.index('needs_fluids'), 'requires', 0.9)
    kg.add_relation(30, symbol_names.index('influenza'), symbol_names.index('needs_medication'), 'requires', 0.7)
    
    kg.add_relation(31, symbol_names.index('covid19'), symbol_names.index('needs_rest'), 'requires', 0.8)
    kg.add_relation(32, symbol_names.index('covid19'), symbol_names.index('needs_fluids'), 'requires', 0.8)
    kg.add_relation(33, symbol_names.index('covid19'), symbol_names.index('needs_monitoring'), 'requires', 0.7)
    kg.add_relation(34, symbol_names.index('covid19'), symbol_names.index('needs_testing'), 'requires', 0.9)
    
    kg.add_relation(35, symbol_names.index('pneumonia'), symbol_names.index('needs_antibiotics'), 'requires', 0.8)
    kg.add_relation(36, symbol_names.index('pneumonia'), symbol_names.index('needs_monitoring'), 'requires', 0.9)
    
    kg.add_relation(37, symbol_names.index('severe_condition'), symbol_names.index('needs_hospitalization'), 'requires', 0.8)
    kg.add_relation(38, symbol_names.index('critical_condition'), symbol_names.index('needs_icu'), 'requires', 0.9)
    kg.add_relation(39, symbol_names.index('critical_condition'), symbol_names.index('needs_ventilation'), 'requires', 0.7)
    
    # Add logical rules
    # COVID-19 diagnostic rules
    kg.add_rule(0, 
               [symbol_names.index('fever'), symbol_names.index('cough'), symbol_names.index('loss_of_taste')],
               symbol_names.index('covid19'),
               0.9)
    
    kg.add_rule(1,
               [symbol_names.index('fever'), symbol_names.index('cough'), symbol_names.index('loss_of_smell')],
               symbol_names.index('covid19'),
               0.9)
    
    # Flu diagnostic rules
    kg.add_rule(2,
               [symbol_names.index('fever'), symbol_names.index('body_aches'), symbol_names.index('fatigue')],
               symbol_names.index('influenza'),
               0.8)
    
    # Common cold diagnostic rules
    kg.add_rule(3,
               [symbol_names.index('runny_nose'), symbol_names.index('sore_throat')],
               symbol_names.index('common_cold'),
               0.7)
    
    # Pneumonia diagnostic rules
    kg.add_rule(4,
               [symbol_names.index('fever'), symbol_names.index('cough'), symbol_names.index('shortness_of_breath')],
               symbol_names.index('pneumonia'),
               0.8)
    
    # Severity rules
    kg.add_rule(5,
               [symbol_names.index('pneumonia'), symbol_names.index('elderly')],
               symbol_names.index('severe_condition'),
               0.8)
    
    kg.add_rule(6,
               [symbol_names.index('covid19'), symbol_names.index('immunocompromised')],
               symbol_names.index('severe_condition'),
               0.8)
    
    kg.add_rule(7,
               [symbol_names.index('pneumonia'), symbol_names.index('severe_shortness_of_breath')],
               symbol_names.index('severe_condition'),
               0.9)
    
    # Treatment rules
    kg.add_rule(8,
               [symbol_names.index('severe_condition')],
               symbol_names.index('needs_hospitalization'),
               0.9)
    
    kg.add_rule(9,
               [symbol_names.index('critical_condition')],
               symbol_names.index('needs_emergency_care'),
               1.0)
    
    return model

def demonstrate_nexus_model():
    """
    Demonstrate the NEXUS Transformer model on sample medical cases
    """
    # Create model
    model = create_medical_nexus_model()
    
    # Sample cases (simplified for demonstration)
    # In a real scenario, these would be properly tokenized text
    sample_cases = [
        # Common cold case
        torch.tensor([[1, 2, 3, 4, 5, 0, 0]]),  # Runny nose, sore throat
        
        # Flu case
        torch.tensor([[6, 7, 8, 9, 10, 0, 0]]),  # Fever, body aches, fatigue
        
        # COVID-19 case
        torch.tensor([[6, 11, 12, 13, 14, 0, 0]]),  # Fever, cough, loss of taste/smell
        
        # Pneumonia case
        torch.tensor([[6, 11, 15, 16, 0, 0, 0]]),  # Fever, cough, shortness of breath
        
        # High-risk case (elderly with COVID symptoms)
        torch.tensor([[6, 11, 14, 17, 0, 0, 0]])  # Fever, cough, loss of taste, elderly
    ]
    
    # Process each case
    case_descriptions = [
        "Common cold symptoms",
        "Flu symptoms",
        "COVID-19 symptoms",
        "Pneumonia symptoms",
        "Elderly patient with COVID symptoms"
    ]
    
    for i, (case, description) in enumerate(zip(sample_cases, case_descriptions)):
        print(f"\n=== Case {i+1}: {description} ===")
        
        # Process case
        result = model.predict(case, task_type='reasoning', risk_level='medium')
        
        # Print results
        print(f"Strategy: {result['strategy']['strategy']}")
        print(f"Explanation:\n{result['explanations'][0]}")
        print("-" * 50)
    
    return model

if __name__ == "__main__":
    # Demonstrate the model
    model = demonstrate_nexus_model() 