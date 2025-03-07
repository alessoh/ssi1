import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd

class NeuralComponent:
    """Neural component of NEXUS that processes patient data."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        """Initialize the neural component with an MLP classifier."""
        self.model = MLPClassifier(
            hidden_layer_sizes=(hidden_dim,),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def train(self, X, y):
        """Train the neural model on the provided data."""
        self.model.fit(X, y)
    
    def process(self, patient_data):
        """Process patient data and return latent representations and predictions."""
        # For a simple toy model, we'll use the MLP's predict_proba as our "latent representation"
        probs = self.model.predict_proba(patient_data)
        predictions = self.model.predict(patient_data)
        
        # In a real implementation, we would extract richer representations
        # Here we'll simulate this by combining raw data with prediction probabilities
        latent_repr = np.hstack([patient_data, probs])
        
        return latent_repr, predictions


class SymbolicComponent:
    """Symbolic component of NEXUS that handles knowledge representation and reasoning."""
    
    def __init__(self):
        """Initialize the knowledge base and reasoning rules."""
        # Simple knowledge base with symptoms and their relation to COVID severity
        self.knowledge_base = {
            # Symptom severity thresholds for COVID categorization
            'fever_threshold': 101.0,  # High fever above 101F is concerning
            'oxygen_threshold': 94,    # O2 below 94% is concerning
            'breathing_threshold': 7,  # Breathing difficulty > 7 is severe
            
            # Comorbidity risk factors (multiplicative effect on risk)
            'risk_factors': {
                'hypertension': 1.5,
                'diabetes': 1.8,
                'heart_disease': 2.0,
                'lung_disease': 2.5,
                'immunocompromised': 3.0
            },
            
            # Treatment recommendations based on severity
            'treatments': {
                'mild': ['rest', 'hydration', 'monitoring', 'acetaminophen'],
                'moderate': ['monoclonal_antibodies', 'antivirals', 'close_monitoring'],
                'severe': ['hospitalization', 'oxygen_therapy', 'dexamethasone', 'remdesivir'],
                'critical': ['icu_admission', 'ventilator', 'dexamethasone', 'remdesivir', 'specialty_consult']
            }
        }
        
        # Define logical inference rules
        self.rules = [
            # Rule 1: High fever suggests at least moderate severity
            lambda data: 'moderate' if data['fever'] > self.knowledge_base['fever_threshold'] else None,
            
            # Rule 2: Low oxygen suggests at least severe
            lambda data: 'severe' if data['oxygen_saturation'] < self.knowledge_base['oxygen_threshold'] else None,
            
            # Rule 3: Severe breathing difficulty suggests critical
            lambda data: 'critical' if data['breathing_difficulty'] > self.knowledge_base['breathing_threshold'] else None,
            
            # Rule 4: Multiple symptoms increase severity
            lambda data: self._evaluate_symptom_combination(data)
        ]
    
    def _evaluate_symptom_combination(self, data):
        """Rule to evaluate the combination of symptoms."""
        symptom_count = (
            (data['fever'] > 100.0) + 
            (data['cough'] > 5) + 
            (data['fatigue'] > 5) +
            (data['breathing_difficulty'] > 3)
        )
        
        if symptom_count >= 3:
            return 'moderate'
        return None
    
    def _calculate_risk_multiplier(self, comorbidities):
        """Calculate risk multiplier based on comorbidities."""
        risk_multiplier = 1.0
        for condition, present in comorbidities.items():
            if present and condition in self.knowledge_base['risk_factors']:
                risk_multiplier *= self.knowledge_base['risk_factors'][condition]
        return risk_multiplier
    
    def reason(self, patient_data):
        """Apply symbolic reasoning to patient data."""
        results = []
        
        for i, patient in enumerate(patient_data):
            patient_dict = {
                'fever': patient[0],
                'cough': patient[1],
                'fatigue': patient[2],
                'breathing_difficulty': patient[3],
                'oxygen_saturation': patient[4],
                'hypertension': patient[5],
                'diabetes': patient[6],
                'heart_disease': patient[7],
                'lung_disease': patient[8],
                'immunocompromised': patient[9]
            }
            
            # Determine base severity using rules
            severity = 'mild'  # Default is mild
            for rule in self.rules:
                rule_result = rule(patient_dict)
                if rule_result is not None:
                    # If a rule suggests higher severity, upgrade it
                    if rule_result == 'moderate' and severity == 'mild':
                        severity = 'moderate'
                    elif rule_result == 'severe' and severity in ['mild', 'moderate']:
                        severity = 'severe'
                    elif rule_result == 'critical':
                        severity = 'critical'
            
            # Calculate comorbidity risk
            comorbidities = {
                'hypertension': bool(patient_dict['hypertension']),
                'diabetes': bool(patient_dict['diabetes']),
                'heart_disease': bool(patient_dict['heart_disease']),
                'lung_disease': bool(patient_dict['lung_disease']),
                'immunocompromised': bool(patient_dict['immunocompromised'])
            }
            risk_multiplier = self._calculate_risk_multiplier(comorbidities)
            
            # Adjust severity based on risk multiplier
            if risk_multiplier > 2.5 and severity == 'mild':
                severity = 'moderate'
            elif risk_multiplier > 3.5 and severity == 'moderate':
                severity = 'severe'
            
            # Get recommended treatments
            treatments = self.knowledge_base['treatments'].get(severity, [])
            
            # Add result to the list
            results.append({
                'patient_id': i + 1,
                'symbolic_severity': severity,
                'risk_multiplier': risk_multiplier,
                'treatments': treatments
            })
        
        return results


class IntegrationMechanism:
    """Integration mechanism that bridges neural and symbolic components."""
    
    def __init__(self, neural_output_dim, symbolic_feature_dim):
        """Initialize the integration mechanism."""
        self.neural_output_dim = neural_output_dim
        self.symbolic_feature_dim = symbolic_feature_dim
        
        # Simple weights for combining neural and symbolic outputs
        # In a real implementation, these would be learned
        self.neural_weight = 0.6
        self.symbolic_weight = 0.4
    
    def neural_to_symbolic(self, neural_repr):
        """Translate neural representations to symbolic knowledge."""
        # In a real implementation, this would be a learned mapping
        # For this toy model, we'll do a simple transformation
        
        # Extract confidence scores for different severities
        # Here we assume the last 4 elements of the neural_repr are prediction probabilities
        severity_scores = neural_repr[:, -4:]  # [mild, moderate, severe, critical]
        
        # Map to symbolic categories based on highest probability
        symbolic_categories = []
        for scores in severity_scores:
            if np.argmax(scores) == 0:
                symbolic_categories.append('mild')
            elif np.argmax(scores) == 1:
                symbolic_categories.append('moderate')
            elif np.argmax(scores) == 2:
                symbolic_categories.append('severe')
            else:
                symbolic_categories.append('critical')
        
        return symbolic_categories
    
    def symbolic_to_neural(self, symbolic_results):
        """Embed symbolic knowledge into neural space."""
        # Convert symbolic severities to numerical values to augment neural features
        severity_map = {'mild': 0, 'moderate': 1, 'severe': 2, 'critical': 3}
        
        symbolic_features = []
        for result in symbolic_results:
            # Create a vector representing symbolic knowledge
            severity_val = severity_map[result['symbolic_severity']]
            risk_val = min(result['risk_multiplier'] / 3.0, 1.0)  # Normalize to [0,1]
            
            # In a real implementation, we would encode the full symbolic knowledge
            # For this toy model, we'll just use these two values
            symbolic_features.append([severity_val, risk_val])
        
        return np.array(symbolic_features)
    
    def joint_reasoning(self, neural_repr, neural_predictions, symbolic_results, symbolic_features):
        """Perform joint neural-symbolic reasoning."""
        integrated_results = []
        
        # Convert neural predictions to the same format as symbolic results
        severity_map = {0: 'mild', 1: 'moderate', 2: 'severe', 3: 'critical'}
        neural_severities = [severity_map[pred] for pred in neural_predictions]
        
        # Integrate neural and symbolic results
        for i, (neural_repr_i, neural_severity, symbolic_result, symbolic_feature) in enumerate(
            zip(neural_repr, neural_severities, symbolic_results, symbolic_features)
        ):
            # Get symbolic severity
            symbolic_severity = symbolic_result['symbolic_severity']
            
            # Check for agreement between neural and symbolic components
            agreement = neural_severity == symbolic_severity
            
            # Determine final severity based on weights
            # If neural and symbolic disagree, use metacognitive control to decide
            # In this toy model, we'll use confidence scores to determine the final decision
            
            # Extract confidence scores from neural representation (example calculation)
            neural_confidence = np.max(neural_repr_i[-4:])  # Max probability as confidence
            
            # Calculate symbolic confidence (example calculation)
            # Here we use risk multiplier as a proxy for confidence
            symbolic_confidence = min(symbolic_result['risk_multiplier'] / 5.0, 1.0)
            
            # Apply metacognitive control
            if neural_confidence > 0.8 and symbolic_confidence < 0.6:
                # Neural component is very confident, symbolic less so
                final_severity = neural_severity
                dominant_component = "Neural"
            elif symbolic_confidence > 0.8 and neural_confidence < 0.6:
                # Symbolic component is very confident, neural less so
                final_severity = symbolic_severity
                dominant_component = "Symbolic"
            else:
                # Weighted combination
                severity_scores = {
                    'mild': 0,
                    'moderate': 1,
                    'severe': 2,
                    'critical': 3
                }
                
                neural_score = severity_scores[neural_severity] * self.neural_weight * neural_confidence
                symbolic_score = severity_scores[symbolic_severity] * self.symbolic_weight * symbolic_confidence
                
                combined_score = neural_score + symbolic_score
                
                # Map back to categorical severity
                if combined_score < 0.5:
                    final_severity = 'mild'
                elif combined_score < 1.5:
                    final_severity = 'moderate'
                elif combined_score < 2.5:
                    final_severity = 'severe'
                else:
                    final_severity = 'critical'
                
                dominant_component = "Integrated"
            
            # Get treatments based on final severity
            treatments = symbolic_result['treatments']
            
            # Create integrated result
            integrated_result = {
                'patient_id': symbolic_result['patient_id'],
                'neural_severity': neural_severity,
                'neural_confidence': neural_confidence,
                'symbolic_severity': symbolic_severity,
                'symbolic_confidence': symbolic_confidence,
                'final_severity': final_severity,
                'dominant_component': dominant_component,
                'treatments': treatments,
                'agreement': agreement
            }
            
            integrated_results.append(integrated_result)
        
        return integrated_results


class NEXUS:
    """Neural-symbolic EXtensible Unified System for medical evaluation."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, symbolic_feature_dim):
        """Initialize NEXUS with neural and symbolic components."""
        self.neural_component = NeuralComponent(input_dim, hidden_dim, output_dim)
        self.symbolic_component = SymbolicComponent()
        self.integration_mechanism = IntegrationMechanism(output_dim, symbolic_feature_dim)
    
    def train(self, X, y):
        """Train the neural component."""
        self.neural_component.train(X, y)
    
    def evaluate(self, patient_data):
        """Evaluate patients using the neural-symbolic architecture."""
        # Step 1: Process data with neural component
        neural_repr, neural_predictions = self.neural_component.process(patient_data)
        
        # Step 2: Apply symbolic reasoning
        symbolic_results = self.symbolic_component.reason(patient_data)
        
        # Step 3: Neural-to-symbolic translation
        neural_symbolic_categories = self.integration_mechanism.neural_to_symbolic(neural_repr)
        
        # Step 4: Symbolic-to-neural embedding
        symbolic_features = self.integration_mechanism.symbolic_to_neural(symbolic_results)
        
        # Step 5: Joint reasoning to produce integrated results
        integrated_results = self.integration_mechanism.joint_reasoning(
            neural_repr, neural_predictions, symbolic_results, symbolic_features
        )
        
        return integrated_results


# Generate synthetic data for 5 COVID-19 patients
def generate_synthetic_data():
    """Generate synthetic data for five COVID-19 patients."""
    # Features: [fever, cough severity (1-10), fatigue (1-10), breathing difficulty (1-10), 
    #            oxygen saturation, hypertension, diabetes, heart disease, lung disease, immunocompromised]
    
    patients = [
        # Patient 1: Mild case
        [99.8, 3, 4, 2, 97, 0, 0, 0, 0, 0],
        
        # Patient 2: Moderate case
        [101.2, 6, 7, 4, 95, 1, 0, 0, 0, 0],
        
        # Patient 3: Severe case
        [102.5, 8, 8, 6, 92, 0, 1, 1, 0, 0],
        
        # Patient 4: Critical case
        [103.1, 9, 9, 9, 88, 1, 1, 1, 1, 0],
        
        # Patient 5: Moderate case with comorbidities (higher risk)
        [100.5, 5, 6, 3, 96, 1, 1, 0, 0, 1]
    ]
    
    # Labels: 0=mild, 1=moderate, 2=severe, 3=critical
    labels = [0, 1, 2, 3, 1]
    
    return np.array(patients), np.array(labels)


# Main function to demonstrate NEXUS
def main():
    # Generate synthetic data
    X, y = generate_synthetic_data()
    
    # Create and train NEXUS
    input_dim = X.shape[1]
    hidden_dim = 8
    output_dim = 4  # 4 severity categories
    symbolic_feature_dim = 2
    
    nexus = NEXUS(input_dim, hidden_dim, output_dim, symbolic_feature_dim)
    nexus.train(X, y)
    
    # Evaluate patients
    results = nexus.evaluate(X)
    
    # Display results as a pandas DataFrame
    df = pd.DataFrame([
        {
            'Patient ID': r['patient_id'],
            'Neural Severity': r['neural_severity'],
            'N-Conf': f"{r['neural_confidence']:.2f}",
            'Symbolic Severity': r['symbolic_severity'],
            'S-Conf': f"{r['symbolic_confidence']:.2f}",
            'Final Severity': r['final_severity'],
            'Dominant': r['dominant_component'],
            'Agreement': 'Yes' if r['agreement'] else 'No',
            'Key Treatments': ', '.join(r['treatments'][:2]) + ('...' if len(r['treatments']) > 2 else '')
        }
        for r in results
    ])
    
    print(df.to_string(index=False))
    
    # Also print detailed treatment plans
    print("\nDetailed Treatment Recommendations:")
    for r in results:
        print(f"Patient {r['patient_id']} ({r['final_severity']}): {', '.join(r['treatments'])}")


if __name__ == "__main__":
    main()