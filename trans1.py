import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

class SymbolicAttention(nn.Module):
    """
    Modified attention mechanism that emphasizes logical relationships 
    and structured knowledge representation
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Relation-aware projection for encoding logical structure
        self.relation_proj = nn.Linear(embed_dim, num_heads)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, logical_mask=None, attention_mask=None):
        batch_size = query.shape[0]
        
        # Project queries, keys and values
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply logical mask if provided - this helps enforce logical constraints
        if logical_mask is not None:
            # Transform logical mask to match attention dimensions
            logical_weights = self.relation_proj(logical_mask).permute(0, 2, 1).unsqueeze(2)
            scores = scores + logical_weights
        
        # Apply standard attention mask if provided
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

class SymbolicTransformerLayer(nn.Module):
    """
    A transformer layer modified to incorporate logical structure and
    symbolic knowledge
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = SymbolicAttention(embed_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network with logical activation
        self.ff_network = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),  # GELU can approximate logical functions
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Additional projection for logical structure
        self.logic_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()  # Sigmoid to act as logical gate
        )
        
    def forward(self, x, logical_mask=None, attention_mask=None):
        # Self-attention block with residual connection
        attn_output, attn_weights = self.self_attn(x, x, x, logical_mask, attention_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward block with logical gating
        ff_output = self.ff_network(x)
        
        # Apply logical gating if logical mask is provided
        if logical_mask is not None:
            logic_gates = self.logic_gate(logical_mask)
            ff_output = ff_output * logic_gates
            
        x = self.norm2(x + ff_output)
        
        return x, attn_weights

class SymbolExtractor(nn.Module):
    """
    Module to extract symbolic representations from neural outputs
    """
    def __init__(self, embed_dim, num_symbols, threshold=0.5):
        super().__init__()
        self.symbol_proj = nn.Linear(embed_dim, num_symbols)
        self.threshold = threshold
        
    def forward(self, x):
        # Project to symbol space
        symbol_logits = self.symbol_proj(x)
        
        # Convert to probabilities
        symbol_probs = torch.sigmoid(symbol_logits)
        
        # Create discrete symbolic representation
        symbols = (symbol_probs > self.threshold).float()
        
        return {
            'logits': symbol_logits,
            'probs': symbol_probs,
            'symbols': symbols
        }

class KnowledgeGraphEmbedding(nn.Module):
    """
    Embeds knowledge graph triples (subject, predicate, object)
    """
    def __init__(self, num_entities, num_relations, embed_dim):
        super().__init__()
        self.entity_embeds = nn.Embedding(num_entities, embed_dim)
        self.relation_embeds = nn.Embedding(num_relations, embed_dim)
        self.triple_proj = nn.Linear(embed_dim * 3, embed_dim)
        
    def forward(self, triples):
        """
        Args:
            triples: (batch_size, 3) tensor of (subject, predicate, object) triples
        """
        subjects = self.entity_embeds(triples[:, 0])
        relations = self.relation_embeds(triples[:, 1])
        objects = self.entity_embeds(triples[:, 2])
        
        # Combine triple representations
        # Using a concatenate and project approach
        triple_embeds = torch.cat([subjects, relations, objects], dim=1)
        triple_embeds = self.triple_proj(triple_embeds)
        
        return triple_embeds

class LogicalConstraintLayer(nn.Module):
    """
    Applies logical constraints to neural representations
    """
    def __init__(self, embed_dim, num_constraints):
        super().__init__()
        self.constraint_weights = nn.Parameter(torch.randn(num_constraints, embed_dim))
        self.constraint_bias = nn.Parameter(torch.zeros(num_constraints))
        
    def forward(self, x):
        # Compute constraint satisfaction scores
        scores = F.linear(x, self.constraint_weights, self.constraint_bias)
        
        # Apply sigmoid to get constraint satisfaction probability
        satisfaction = torch.sigmoid(scores)
        
        # Create mask that enforces constraints
        # (gradient will flow back to model to learn to satisfy constraints)
        constraint_mask = satisfaction.unsqueeze(-1).expand_as(x)
        
        # Apply mask to input (soft constraint)
        logically_consistent_x = x * constraint_mask
        
        return logically_consistent_x, satisfaction

class NeuroSymbolicTransformer(nn.Module):
    """
    A transformer model that integrates neural processing with symbolic reasoning
    """
    def __init__(self, 
                 vocab_size, 
                 embed_dim, 
                 num_heads, 
                 ff_dim, 
                 num_layers, 
                 num_symbols,
                 num_entities,
                 num_relations,
                 num_constraints,
                 max_seq_len=512,
                 dropout=0.1):
        super().__init__()
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        
        # Knowledge graph embedding
        self.kg_embedding = KnowledgeGraphEmbedding(num_entities, num_relations, embed_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            SymbolicTransformerLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Logical constraint layer
        self.logical_constraint = LogicalConstraintLayer(embed_dim, num_constraints)
        
        # Symbol extractor
        self.symbol_extractor = SymbolExtractor(embed_dim, num_symbols)
        
        # Final projection layer
        self.final_proj = nn.Linear(embed_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        
        # Initialize positional embeddings
        nn.init.normal_(self.pos_embedding, mean=0, std=0.02)
        
    def integrate_kg_knowledge(self, x, kg_triples=None):
        """
        Integrates knowledge graph information with input embeddings
        """
        if kg_triples is not None:
            # Ensure kg_triples is properly formatted
            if isinstance(kg_triples, list):
                kg_triples = torch.tensor(kg_triples, dtype=torch.long)
                
            if len(kg_triples.shape) == 1:
                # If we have a single triple, add batch dimension
                kg_triples = kg_triples.unsqueeze(0)
                
            # Get knowledge graph embeddings
            kg_embeds = self.kg_embedding(kg_triples)
            
            # Use simple addition to integrate KG knowledge
            # In a real implementation, this would be more sophisticated
            batch_size, seq_len, _ = x.size()
            
            # Expand kg embeddings to match sequence length
            kg_embeds = kg_embeds.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Add KG knowledge to input embeddings
            x = x + kg_embeds
            
        return x
        
    def forward(self, 
                input_ids, 
                attention_mask=None, 
                kg_triples=None, 
                logical_constraints=None):
        
        seq_length = input_ids.size(1)
        
        # Get token embeddings
        x = self.token_embedding(input_ids)
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :seq_length, :]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Integrate knowledge graph information
        x = self.integrate_kg_knowledge(x, kg_triples)
        
        # Create logical mask from constraints if provided
        logical_mask = None
        if logical_constraints is not None:
            # Apply logical constraints
            x, satisfaction = self.logical_constraint(x)
            # Create a mask based on constraint satisfaction
            logical_mask = satisfaction.unsqueeze(-1).expand_as(x)
        
        # Apply transformer layers
        all_attentions = []
        for layer in self.layers:
            x, attn_weights = layer(x, logical_mask, attention_mask)
            all_attentions.append(attn_weights)
        
        # Extract symbolic representation
        symbolic_output = self.symbol_extractor(x)
        
        # Project to vocabulary for token generation
        logits = self.final_proj(x)
        
        return {
            'logits': logits,  # For language modeling
            'symbol_logits': symbolic_output['logits'],  # Symbolic representation logits
            'symbols': symbolic_output['symbols'],  # Discrete symbolic representation
            'embeddings': x,  # Final embeddings for downstream tasks
            'attentions': all_attentions  # Attention weights for analysis
        }

class SymbolicReasoningEngine:
    """
    A simple symbolic reasoning engine that applies rules to symbolic representations
    """
    def __init__(self, symbol_names, rules=None):
        self.symbol_names = symbol_names
        self.rules = rules or []
        
    def add_rule(self, premise, conclusion):
        """
        Add a rule of the form "if premise then conclusion"
        
        Args:
            premise: List of indices of symbols that must be true
            conclusion: List of indices of symbols that will be inferred
        """
        self.rules.append((premise, conclusion))
        
    def reason(self, symbols):
        """
        Apply logical rules to derive new symbols
        
        Args:
            symbols: Binary tensor of shape [batch_size, num_symbols]
            
        Returns:
            Updated symbols after reasoning
        """
        # Extract the shape and ensure it's properly formatted
        if len(symbols.shape) == 3:  # [batch_size, seq_len, num_symbols]
            # Use only the first token's symbols for simplicity
            symbols = symbols[:, 0, :]
        
        # Convert to numpy for easier manipulation
        batch_size, num_symbols = symbols.shape
        symbols_np = symbols.cpu().numpy()
        
        # Apply rules until no more changes
        changes = True
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while changes and iteration < max_iterations:
            changes = False
            iteration += 1
            
            for premise, conclusion in self.rules:
                # Check if all premise symbols are active
                if not premise:  # Skip empty premises
                    continue
                    
                # Handle single premises differently from multiple premises
                if len(premise) == 1:
                    premise_satisfied = symbols_np[:, premise[0]] > 0.5
                else:
                    premise_satisfied = np.all(symbols_np[:, premise] > 0.5, axis=1)
                
                # For each satisfied premise, activate conclusion symbols
                for b in range(batch_size):
                    if premise_satisfied[b] if np.isscalar(premise_satisfied) else premise_satisfied[b]:
                        for c in conclusion:
                            if symbols_np[b, c] < 0.5:
                                symbols_np[b, c] = 1.0
                                changes = True
        
        # Convert back to tensor
        return torch.tensor(symbols_np, device=symbols.device)
    
    def explain(self, symbols, index):
        """
        Explain why a particular symbol is activated for a specific example
        
        Args:
            symbols: Binary tensor of shape [batch_size, num_symbols]
            index: Index of the example to explain
            
        Returns:
            Explanation as a string
        """
        # Handle different tensor shapes
        if len(symbols.shape) == 3:  # [batch_size, seq_len, num_symbols]
            symbols = symbols[:, 0, :]
            
        symbols_np = symbols[index].cpu().numpy()
        active_symbols = [self.symbol_names[i] for i in range(len(symbols_np)) if symbols_np[i] > 0.5]
        
        explanations = []
        for premise, conclusion in self.rules:
            if not premise:  # Skip empty premises
                continue
                
            premise_symbols = [self.symbol_names[i] for i in premise]
            conclusion_symbols = [self.symbol_names[i] for i in conclusion]
            
            # Check if all premise symbols are active
            if all(self.symbol_names[i] in active_symbols for i in premise):
                # Check if any conclusion symbols are active and were derived (not in input)
                derived = [s for s in conclusion_symbols if s in active_symbols]
                if derived:
                    explanations.append(f"Because {' AND '.join(premise_symbols)}, we can conclude {' AND '.join(derived)}")
        
        if not explanations:
            return "No applicable rules found for the active symbols."
        
        return "\n".join(explanations)

class NeuroSymbolicSystem:
    """
    Complete neuro-symbolic system that combines neural processing with symbolic reasoning
    """
    def __init__(self, neural_model, symbolic_engine, tokenizer=None):
        self.neural_model = neural_model
        self.symbolic_engine = symbolic_engine
        self.tokenizer = tokenizer
        
    def process(self, input_text=None, input_ids=None, kg_triples=None):
        """
        Process input through the neural-symbolic system
        
        Args:
            input_text: Text input (will be tokenized if tokenizer is provided)
            input_ids: Direct token IDs input (alternative to input_text)
            kg_triples: Knowledge graph triples to incorporate
            
        Returns:
            Dictionary with neural outputs, symbolic outputs, and reasoning
        """
        # Tokenize input if needed
        if input_ids is None and input_text is not None and self.tokenizer is not None:
            input_ids = self.tokenizer(input_text, return_tensors="pt")["input_ids"]
        
        if input_ids is None:
            raise ValueError("Either input_text with tokenizer or input_ids must be provided")
            
        # Generate attention mask (1 for all tokens)
        attention_mask = torch.ones_like(input_ids)
        
        # Process through neural model
        neural_outputs = self.neural_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            kg_triples=kg_triples
        )
        
        # Extract symbolic representations
        symbols = neural_outputs['symbols']
        
        # Apply symbolic reasoning
        reasoned_symbols = self.symbolic_engine.reason(symbols)
        
        # Generate text outputs if tokenizer is available
        text_output = None
        if self.tokenizer is not None:
            logits = neural_outputs['logits']
            output_ids = torch.argmax(logits, dim=-1)
            text_output = self.tokenizer.decode(output_ids[0])
        
        # Generate explanations for each example
        explanations = []
        for i in range(symbols.shape[0]):
            explanations.append(self.symbolic_engine.explain(reasoned_symbols, i))
        
        return {
            'neural_outputs': neural_outputs,
            'symbols': symbols,
            'reasoned_symbols': reasoned_symbols,
            'text_output': text_output,
            'explanations': explanations
        }

# Simple tokenizer for demonstration purposes
class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        self.id_to_token = {i: token for i, token in enumerate(vocab)}
        
    def encode(self, text):
        # Extremely simplified tokenization (just splits by space)
        tokens = text.lower().split()
        return [self.token_to_id.get(token, self.token_to_id["<unk>"]) for token in tokens]
    
    def decode(self, ids):
        return " ".join([self.id_to_token.get(id, "<unk>") for id in ids])
    
    def __call__(self, text, return_tensors=None):
        ids = self.encode(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}

# Sample medical domain implementation
def create_medical_neuro_symbolic_system():
    # Define vocabulary
    vocab = ["<unk>", "patient", "has", "fever", "cough", "shortness", "of", "breath", "fatigue", 
             "mild", "moderate", "severe", "diagnosis", "is", "flu", "pneumonia", "covid", 
             "needs", "hospitalization", "oxygen", "antibiotics", "treatment", "with", "concern", 
             "for", "elderly", "immunocompromised", "<pad>"]
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab)
    
    # Define medical symbols
    medical_symbols = [
        "has_fever", "has_high_fever", "has_cough", "has_severe_cough", 
        "has_shortness_of_breath", "has_severe_sob", "has_fatigue",
        "is_elderly", "is_immunocompromised",
        "condition_mild", "condition_pneumonia", "condition_severe",
        "need_hospitalization", "need_oxygen", "need_antibiotics"
    ]
    
    # Create neural model
    model = NeuroSymbolicTransformer(
        vocab_size=len(vocab),
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
        num_layers=2,
        num_symbols=len(medical_symbols),
        num_entities=100,  # Simplified KG with just 100 entities
        num_relations=10,  # 10 relation types
        num_constraints=5,  # 5 logical constraints
        max_seq_len=50
    )
    
    # Create symbolic reasoning engine
    reasoning_engine = SymbolicReasoningEngine(medical_symbols)
    
    # Add some basic medical reasoning rules
    # Rule format: (premises, conclusions)
    reasoning_engine.add_rule(
        premise=[0, 3],  # has_fever AND has_severe_cough
        conclusion=[11]  # condition_severe
    )
    reasoning_engine.add_rule(
        premise=[0, 4],  # has_fever AND has_shortness_of_breath
        conclusion=[10]  # condition_pneumonia
    )
    reasoning_engine.add_rule(
        premise=[11],  # condition_severe
        conclusion=[12, 13]  # need_hospitalization AND need_oxygen
    )
    reasoning_engine.add_rule(
        premise=[10],  # condition_pneumonia
        conclusion=[14]  # need_antibiotics
    )
    reasoning_engine.add_rule(
        premise=[8, 10],  # is_immunocompromised AND condition_pneumonia
        conclusion=[12]  # need_hospitalization
    )
    
    # Create complete system
    system = NeuroSymbolicSystem(model, reasoning_engine, tokenizer)
    
    return system, medical_symbols

# Define simple knowledge graph triples for the medical domain
def create_medical_kg_triples():
    # Format: (subject_id, relation_id, object_id)
    # Example mappings (simplified):
    # Entity IDs: 0=fever, 1=cough, 2=pneumonia, 3=flu, 4=covid, 5=elderly
    # Relation IDs: 0=is_symptom_of, 1=increases_risk_for, 2=requires
    
    return torch.tensor([
        [0, 0, 3],  # fever is_symptom_of flu
        [0, 0, 4],  # fever is_symptom_of covid
        [1, 0, 3],  # cough is_symptom_of flu
        [1, 0, 2],  # cough is_symptom_of pneumonia
        [5, 1, 2]   # elderly increases_risk_for pneumonia
    ])

# Demo function to run the system with sample inputs
def run_neuro_symbolic_demo():
    print("Initializing Neural-Symbolic Medical System...")
    system, symbol_names = create_medical_neuro_symbolic_system()
    kg_triples = create_medical_kg_triples()
    
    # Sample inputs
    test_inputs = [
        "patient has fever cough and fatigue",
        "patient has severe cough shortness of breath",
        "elderly patient has fever and shortness of breath",
        "immunocompromised patient has cough shortness of breath"
    ]
    
    print("\nRunning inference on sample inputs...\n")
    
    for i, input_text in enumerate(test_inputs):
        print(f"Input {i+1}: \"{input_text}\"")
        
        # Process through the neural-symbolic system
        result = system.process(input_text=input_text, kg_triples=kg_triples)
        
        # Print the symbolic output and explanation
        symbols = result['reasoned_symbols'][0].cpu().numpy()
        active_symbols = [f"{symbol_names[j]}" for j in range(len(symbols)) if symbols[j] > 0.5]
        
        print(f"Detected conditions and symptoms:")
        for symbol in active_symbols:
            print(f"  - {symbol}")
        
        print("\nReasoning explanation:")
        print(result['explanations'][0])
        print("-" * 50)
    
    return system

# Run the demo if this file is executed directly
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Enable debugging - trace more information
    debug = True
    
    # Print PyTorch version for debugging
    print(f"PyTorch version: {torch.__version__}")
    
    try:
        system = run_neuro_symbolic_demo()
        print("Demo completed successfully!")
    except Exception as e:
        print(f"Error occurred: {e}")
        
        # Print more detailed error information in debug mode
        if debug:
            import traceback
            print("\nDetailed error information:")
            traceback.print_exc()
            
            # Print more information about the state at the point of error
            print("\nAttempting to print additional debug information...")
            try:
                # Initialize the system again to see where it fails
                print("Re-initializing system...")
                system_test, _ = create_medical_neuro_symbolic_system()
                
                # Test the shape of symbols to identify issues
                print("Testing symbolic processing...")
                # Create dummy inputs
                dummy_input = torch.ones((1, 5), dtype=torch.long)  # Simple 1 example, 5 tokens
                dummy_kg = torch.tensor([[0, 0, 1]])  # Simple KG triple
                
                # Process through neural model only
                print("Processing through neural model...")
                with torch.no_grad():
                    outputs = system_test.neural_model(dummy_input, kg_triples=dummy_kg)
                
                # Print shape information
                print(f"Symbol shape: {outputs['symbols'].shape}")
                
                if len(outputs['symbols'].shape) != 2:
                    print("Symbols tensor doesn't have expected shape [batch_size, num_symbols]")
                    print("This is likely causing the 'too many values to unpack' error")
            except Exception as debug_error:
                print(f"Debug diagnostics failed: {debug_error}")
    
    print("Script execution complete.")