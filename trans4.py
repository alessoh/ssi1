import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
import random

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
        self.symbol_to_neural_matrix = torch.randn(num_symbols, input_dim)
        
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
        neural_repr = torch.matmul(symbolic_activations, self.symbol_to_neural_matrix)
        
        return neural_repr

class NexusToyModel:
    """
    Improved NEXUS toy model with enhanced explanation capabilities
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
    
    def process_patient_symptoms(self, symptoms):
        """
        Process a patient case based on symptom names
        
        Args:
            symptoms: List of symptom names
            
        Returns:
            Inferred symbols and reasoning steps
        """
        # Convert symptom names to symbol IDs
        symptom_ids = []
        for symptom in symptoms:
            if symptom in self.symbol_to_id:
                symptom_ids.append(self.symbol_to_id[symptom])
            else:
                print(f"Warning: Unknown symptom '{symptom}'")
        
        # Apply reasoning
        inferred_symbols, reasoning_steps = self.knowledge_graph.reason(symptom_ids)
        
        return inferred_symbols, reasoning_steps
    
    def explain(self, inferred_symbols, reasoning_steps):
        """Generate improved explanation for the inference process"""
        # Convert symbol IDs to names
        active_symbols = [s for s in inferred_symbols if s < len(self.symbol_names)]
        active_names = [self.symbol_names[i] for i in active_symbols]
        
        # Categorize identified concepts
        symptoms = []
        conditions = []
        treatments = []
        risk_factors = []
        severity = []
        
        # Define which symbols belong to which category
        symptom_list = ['fever', 'high_fever', 'cough', 'severe_cough', 'fatigue', 
                        'shortness_of_breath', 'severe_shortness_of_breath', 'sore_throat', 
                        'runny_nose', 'body_aches', 'headache', 'loss_of_taste', 'loss_of_smell']
        
        condition_list = ['common_cold', 'influenza', 'covid19', 'pneumonia', 'bronchitis']
        
        treatment_list = ['rest', 'fluids', 'medication', 'antibiotics', 'antiviral', 
                          'hospitalization', 'oxygen', 'ventilator', 'monitoring']
        
        risk_factor_list = ['elderly', 'immunocompromised', 'hypertension', 'diabetes', 'obesity']
        
        severity_list = ['mild', 'moderate', 'severe', 'critical']
        
        for name in active_names:
            if name in symptom_list:
                symptoms.append(name)
            elif name in condition_list:
                conditions.append(name)
            elif name in treatment_list:
                treatments.append(name)
            elif name in risk_factor_list:
                risk_factors.append(name)
            elif name in severity_list:
                severity.append(name)
        
        # Build a structured explanation
        explanation = ["=== Medical Analysis Summary ===\n"]
        
        # Symptoms section
        if symptoms:
            explanation.append("IDENTIFIED SYMPTOMS:")
            for symptom in symptoms:
                explanation.append(f"  • {symptom.replace('_', ' ').capitalize()}")
            explanation.append("")
        else:
            explanation.append("NO SYMPTOMS IDENTIFIED\n")
        
        # Risk factors section
        if risk_factors:
            explanation.append("RISK FACTORS:")
            for factor in risk_factors:
                explanation.append(f"  • {factor.replace('_', ' ').capitalize()}")
            explanation.append("")
        
        # Severity section
        if severity:
            explanation.append("ASSESSED SEVERITY:")
            for sev in severity:
                explanation.append(f"  • {sev.capitalize()}")
            explanation.append("")
        
        # Conditions section
        if conditions:
            explanation.append("POTENTIAL CONDITIONS:")
            for condition in conditions:
                # Find reasoning for this condition
                reason = ""
                if condition in self.symbol_to_id and self.symbol_to_id[condition] in reasoning_steps:
                    step = reasoning_steps[self.symbol_to_id[condition]]
                    reason = f" (Based on: {step})"
                
                # Format condition name nicely
                if condition == "covid19":
                    formatted_condition = "COVID-19"
                else:
                    formatted_condition = condition.replace('_', ' ').capitalize()
                    
                explanation.append(f"  • {formatted_condition}{reason}")
            explanation.append("")
        else:
            explanation.append("NO CONDITIONS IDENTIFIED\n")
        
        # Treatments section
        if treatments:
            explanation.append("RECOMMENDED ACTIONS:")
            for treatment in treatments:
                # Find reasoning for this treatment
                reason = ""
                if treatment in self.symbol_to_id and self.symbol_to_id[treatment] in reasoning_steps:
                    step = reasoning_steps[self.symbol_to_id[treatment]]
                    
                    # Extract the condition that led to this treatment
                    if "Rule: IF" in step:
                        condition_part = step.split("Rule: IF ")[1].split(" THEN")[0]
                        # Format condition name nicely
                        if condition_part == "covid19":
                            condition_part = "COVID-19"
                        else:
                            condition_part = condition_part.replace('_', ' ').capitalize()
                        reason = f" (For: {condition_part})"
                
                # Format treatment name nicely
                formatted_treatment = treatment.replace('_', ' ').capitalize()
                explanation.append(f"  • {formatted_treatment}{reason}")
            explanation.append("")
        else:
            explanation.append("NO SPECIFIC ACTIONS RECOMMENDED\n")
        
        # Detailed reasoning section
        explanation.append("DETAILED REASONING:")
        if reasoning_steps:
            # Group reasoning steps by type
            symptom_relations = []
            risk_relations = []
            rule_applications = []
            
            for symbol_id, step in reasoning_steps.items():
                if "--symptom_of-->" in step:
                    symptom_relations.append(step)
                elif "--increases_risk-->" in step:
                    risk_relations.append(step)
                elif "Rule: IF" in step:
                    rule_applications.append(step)
            
            # Symptom relationships
            if symptom_relations:
                explanation.append("  Symptom Relationships:")
                for step in symptom_relations:
                    # Format the relationship more clearly
                    parts = step.split(" --symptom_of--> ")
                    if len(parts) == 2:
                        symptom, condition = parts
                        # Format condition name nicely
                        if condition == "covid19":
                            condition = "COVID-19"
                        else:
                            condition = condition.replace('_', ' ').capitalize()
                        explanation.append(f"    → {symptom.replace('_', ' ').capitalize()} is a symptom of {condition}")
            
            # Risk factor relationships
            if risk_relations:
                explanation.append("  Risk Factor Analysis:")
                for step in risk_relations:
                    parts = step.split(" --increases_risk--> ")
                    if len(parts) == 2:
                        factor, condition = parts
                        # Format names nicely
                        if condition == "covid19":
                            condition = "COVID-19"
                        else:
                            condition = condition.replace('_', ' ').capitalize()
                        factor = factor.replace('_', ' ').capitalize()
                        explanation.append(f"    → {factor} increases risk for {condition}")
            
            # Rule applications
            if rule_applications:
                explanation.append("  Medical Rules Applied:")
                for step in rule_applications:
                    # Format the rule more clearly
                    if "Rule: IF" in step and "THEN" in step:
                        rule_parts = step.split("Rule: IF ")[1].split(" THEN ")
                        if len(rule_parts) == 2:
                            condition, action = rule_parts
                            # Format names nicely
                            if "covid19" in condition:
                                condition = condition.replace("covid19", "COVID-19")
                            condition = condition.replace('_', ' ').replace(' AND ', ' and ')
                            action = action.replace('_', ' ').capitalize()
                            explanation.append(f"    → If a patient has {condition}, then {action} is recommended")
        else:
            explanation.append("  No detailed reasoning steps available.")
        
        # Add confidence assessment
        explanation.append("\nCONFIDENCE ASSESSMENT:")
        
        # Determine confidence based on factors like number of symptoms and risk factors
        num_symptoms = len(symptoms)
        num_risk_factors = len(risk_factors)
        has_severe_symptoms = any(s.startswith('severe_') for s in symptoms)
        
        if num_symptoms >= 3 or (num_symptoms >= 2 and has_severe_symptoms):
            confidence = "HIGH"
        elif num_symptoms >= 2 or has_severe_symptoms or num_risk_factors >= 1:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
            
        explanation.append(f"  • The model's confidence in this analysis is {confidence}")
        
        if confidence != "HIGH":
            explanation.append("  • Additional symptoms would help refine this assessment")
        
        # Add disclaimer
        explanation.append("\nDISCLAIMER: This is an AI-generated analysis for demonstration purposes only.")
        explanation.append("Always consult with a qualified healthcare professional for medical advice.")
        
        return "\n".join(explanation)


# Create an enhanced medical knowledge model
def create_enhanced_medical_model():
    # Model parameters
    embed_dim = 64
    num_heads = 4
    ff_dim = 128
    num_layers = 2
    
    # Define medical symbols (expanded set)
    symbols = [
        # Symptoms
        'fever', 'high_fever', 'cough', 'severe_cough', 'fatigue', 'severe_fatigue',
        'shortness_of_breath', 'severe_shortness_of_breath', 'sore_throat', 
        'runny_nose', 'body_aches', 'headache', 'loss_of_taste', 'loss_of_smell',
        
        # Risk factors
        'elderly', 'immunocompromised', 'hypertension', 'diabetes', 'obesity',
        
        # Conditions
        'common_cold', 'influenza', 'covid19', 'pneumonia', 'bronchitis',
        
        # Severity
        'mild', 'moderate', 'severe', 'critical',
        
        # Treatments
        'rest', 'fluids', 'medication', 'antibiotics', 'antiviral',
        'monitoring', 'hospitalization', 'oxygen', 'ventilator'
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
    symbol_to_id = model.symbol_to_id
    
    # Add entities
    for name, idx in symbol_to_id.items():
        kg.add_entity(idx, name)
    
    # Add symptom-to-condition relations
    kg.add_relation(symbol_to_id['fever'], "symptom_of", symbol_to_id['common_cold'], 0.5)
    kg.add_relation(symbol_to_id['fever'], "symptom_of", symbol_to_id['influenza'], 0.8)
    kg.add_relation(symbol_to_id['fever'], "symptom_of", symbol_to_id['covid19'], 0.7)
    kg.add_relation(symbol_to_id['fever'], "symptom_of", symbol_to_id['pneumonia'], 0.7)
    
    kg.add_relation(symbol_to_id['high_fever'], "symptom_of", symbol_to_id['influenza'], 0.9)
    kg.add_relation(symbol_to_id['high_fever'], "symptom_of", symbol_to_id['pneumonia'], 0.8)
    kg.add_relation(symbol_to_id['high_fever'], "symptom_of", symbol_to_id['covid19'], 0.6)
    
    kg.add_relation(symbol_to_id['cough'], "symptom_of", symbol_to_id['common_cold'], 0.7)
    kg.add_relation(symbol_to_id['cough'], "symptom_of", symbol_to_id['influenza'], 0.7)
    kg.add_relation(symbol_to_id['cough'], "symptom_of", symbol_to_id['covid19'], 0.8)
    kg.add_relation(symbol_to_id['cough'], "symptom_of", symbol_to_id['bronchitis'], 0.9)
    kg.add_relation(symbol_to_id['cough'], "symptom_of", symbol_to_id['pneumonia'], 0.8)
    
    kg.add_relation(symbol_to_id['severe_cough'], "symptom_of", symbol_to_id['bronchitis'], 0.9)
    kg.add_relation(symbol_to_id['severe_cough'], "symptom_of", symbol_to_id['pneumonia'], 0.8)
    
    kg.add_relation(symbol_to_id['fatigue'], "symptom_of", symbol_to_id['common_cold'], 0.5)
    kg.add_relation(symbol_to_id['fatigue'], "symptom_of", symbol_to_id['influenza'], 0.8)
    kg.add_relation(symbol_to_id['fatigue'], "symptom_of", symbol_to_id['covid19'], 0.7)
    
    kg.add_relation(symbol_to_id['shortness_of_breath'], "symptom_of", symbol_to_id['pneumonia'], 0.9)
    kg.add_relation(symbol_to_id['shortness_of_breath'], "symptom_of", symbol_to_id['covid19'], 0.7)
    
    kg.add_relation(symbol_to_id['severe_shortness_of_breath'], "symptom_of", symbol_to_id['pneumonia'], 0.95)
    kg.add_relation(symbol_to_id['severe_shortness_of_breath'], "symptom_of", symbol_to_id['covid19'], 0.8)
    
    kg.add_relation(symbol_to_id['sore_throat'], "symptom_of", symbol_to_id['common_cold'], 0.8)
    kg.add_relation(symbol_to_id['sore_throat'], "symptom_of", symbol_to_id['influenza'], 0.4)
    kg.add_relation(symbol_to_id['sore_throat'], "symptom_of", symbol_to_id['covid19'], 0.5)
    
    kg.add_relation(symbol_to_id['runny_nose'], "symptom_of", symbol_to_id['common_cold'], 0.9)
    kg.add_relation(symbol_to_id['runny_nose'], "symptom_of", symbol_to_id['influenza'], 0.3)
    
    kg.add_relation(symbol_to_id['body_aches'], "symptom_of", symbol_to_id['influenza'], 0.9)
    kg.add_relation(symbol_to_id['body_aches'], "symptom_of", symbol_to_id['covid19'], 0.6)
    
    kg.add_relation(symbol_to_id['headache'], "symptom_of", symbol_to_id['common_cold'], 0.6)
    kg.add_relation(symbol_to_id['headache'], "symptom_of", symbol_to_id['influenza'], 0.7)
    kg.add_relation(symbol_to_id['headache'], "symptom_of", symbol_to_id['covid19'], 0.5)
    
    kg.add_relation(symbol_to_id['loss_of_taste'], "symptom_of", symbol_to_id['covid19'], 0.9)
    kg.add_relation(symbol_to_id['loss_of_smell'], "symptom_of", symbol_to_id['covid19'], 0.9)
    
    # Add risk factor relations
    kg.add_relation(symbol_to_id['elderly'], "increases_risk", symbol_to_id['severe'], 0.8)
    kg.add_relation(symbol_to_id['immunocompromised'], "increases_risk", symbol_to_id['severe'], 0.9)
    kg.add_relation(symbol_to_id['diabetes'], "increases_risk", symbol_to_id['severe'], 0.7)
    kg.add_relation(symbol_to_id['hypertension'], "increases_risk", symbol_to_id['severe'], 0.7)
    kg.add_relation(symbol_to_id['obesity'], "increases_risk", symbol_to_id['severe'], 0.7)
    
    # Add condition severity assessment rules
    kg.add_rule([symbol_to_id['common_cold']], symbol_to_id['mild'])
    kg.add_rule([symbol_to_id['influenza']], symbol_to_id['moderate'])
    
    kg.add_rule([symbol_to_id['covid19']], symbol_to_id['moderate'])
    kg.add_rule([symbol_to_id['covid19'], symbol_to_id['elderly']], symbol_to_id['severe'])
    kg.add_rule([symbol_to_id['covid19'], symbol_to_id['immunocompromised']], symbol_to_id['severe'])
    kg.add_rule([symbol_to_id['covid19'], symbol_to_id['diabetes']], symbol_to_id['severe'])
    kg.add_rule([symbol_to_id['covid19'], symbol_to_id['severe_shortness_of_breath']], symbol_to_id['severe'])
    
    kg.add_rule([symbol_to_id['pneumonia']], symbol_to_id['severe'])
    kg.add_rule([symbol_to_id['pneumonia'], symbol_to_id['elderly']], symbol_to_id['critical'])
    kg.add_rule([symbol_to_id['pneumonia'], symbol_to_id['immunocompromised']], symbol_to_id['critical'])
    
    # Add treatment rules based on conditions
    kg.add_rule([symbol_to_id['common_cold']], symbol_to_id['rest'])
    kg.add_rule([symbol_to_id['common_cold']], symbol_to_id['fluids'])
    
    kg.add_rule([symbol_to_id['influenza']], symbol_to_id['rest'])
    kg.add_rule([symbol_to_id['influenza']], symbol_to_id['fluids'])
    kg.add_rule([symbol_to_id['influenza']], symbol_to_id['medication'])
    
    kg.add_rule([symbol_to_id['bronchitis']], symbol_to_id['rest'])
    kg.add_rule([symbol_to_id['bronchitis']], symbol_to_id['fluids'])
    kg.add_rule([symbol_to_id['bronchitis']], symbol_to_id['medication'])
    
    kg.add_rule([symbol_to_id['covid19']], symbol_to_id['rest'])
    kg.add_rule([symbol_to_id['covid19']], symbol_to_id['fluids'])
    kg.add_rule([symbol_to_id['covid19']], symbol_to_id['monitoring'])
    kg.add_rule([symbol_to_id['covid19'], symbol_to_id['moderate']], symbol_to_id['antiviral'])
    
    kg.add_rule([symbol_to_id['pneumonia']], symbol_to_id['antibiotics'])
    
    # Add treatment rules based on severity
    kg.add_rule([symbol_to_id['moderate']], symbol_to_id['monitoring'])
    kg.add_rule([symbol_to_id['severe']], symbol_to_id['hospitalization'])
    kg.add_rule([symbol_to_id['severe']], symbol_to_id['monitoring'])
    kg.add_rule([symbol_to_id['critical']], symbol_to_id['hospitalization'])
    kg.add_rule([symbol_to_id['critical']], symbol_to_id['oxygen'])
    kg.add_rule([symbol_to_id['critical'], symbol_to_id['severe_shortness_of_breath']], symbol_to_id['ventilator'])
    
    return model

# Define 10 synthetic patient cases
def define_patient_cases():
    cases = [
        {
            "id": 1,
            "description": "Common Cold",
            "symptoms": ["runny_nose", "sore_throat", "cough"],
            "risk_factors": []
        },
        {
            "id": 2,
            "description": "Influenza",
            "symptoms": ["fever", "body_aches", "fatigue", "headache"],
            "risk_factors": []
        },
        {
            "id": 3,
            "description": "COVID-19 (Mild)",
            "symptoms": ["fever", "cough", "fatigue", "loss_of_taste"],
            "risk_factors": []
        },
        {
            "id": 4,
            "description": "COVID-19 (Severe in Elderly)",
            "symptoms": ["fever", "cough", "shortness_of_breath", "loss_of_taste"],
            "risk_factors": ["elderly"]
        },
        {
            "id": 5,
            "description": "Pneumonia",
            "symptoms": ["high_fever", "severe_cough", "shortness_of_breath"],
            "risk_factors": []
        },
        {
            "id": 6,
            "description": "Severe Pneumonia in Immunocompromised Patient",
            "symptoms": ["high_fever", "severe_cough", "severe_shortness_of_breath"],
            "risk_factors": ["immunocompromised"]
        },
        {
            "id": 7,
            "description": "Bronchitis",
            "symptoms": ["cough", "severe_cough", "fatigue"],
            "risk_factors": []
        },
        {
            "id": 8,
            "description": "Mixed Symptoms",
            "symptoms": ["fever", "runny_nose", "cough", "headache"],
            "risk_factors": []
        },
        {
            "id": 9,
            "description": "COVID-19 with Diabetes",
            "symptoms": ["fever", "cough", "shortness_of_breath", "loss_of_smell"],
            "risk_factors": ["diabetes"]
        },
        {
            "id": 10,
            "description": "Critical Pneumonia in Elderly",
            "symptoms": ["high_fever", "severe_cough", "severe_shortness_of_breath"],
            "risk_factors": ["elderly", "hypertension"]
        }
    ]
    return cases

# Test the model with the synthetic cases
def test_patient_cases():
    # Create the enhanced model
    model = create_enhanced_medical_model()
    
    # Get the patient cases
    cases = define_patient_cases()
    
    print("===== NEXUS Medical Model: Synthetic Patient Case Analysis =====\n")
    
    # Process each case
    for case in cases:
        print(f"CASE #{case['id']}: {case['description']}")
        print("-" * 80)
        
        # Combine symptoms and risk factors for processing
        all_inputs = case['symptoms'] + case['risk_factors']
        
        # Process the case
        inferred_symbols, reasoning_steps = model.process_patient_symptoms(all_inputs)
        
        # Generate explanation
        explanation = model.explain(inferred_symbols, reasoning_steps)
        
        # Print the explanation
        print(explanation)
        print("\n" + "=" * 80 + "\n")
    
    return model, cases

# Run the test
if __name__ == "__main__":
    test_patient_cases()