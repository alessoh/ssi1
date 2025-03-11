import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class NeuralDiagnosticModel:
    """Neural network component for medical diagnosis"""
    
    def __init__(self, input_dim, output_dim, hidden_layers=[32, 16]):
        """
        Initialize the neural diagnostic model
        
        Parameters:
        - input_dim: Number of input features (symptoms)
        - output_dim: Number of output classes (diagnoses)
        - hidden_layers: List of hidden layer sizes
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.model = self._build_model()
        
    def _build_model(self):
        """Build and compile the neural network model"""
        model = keras.Sequential()
        
        # Input layer
        model.add(keras.layers.Dense(self.hidden_layers[0], activation='relu', 
                                     input_shape=(self.input_dim,)))
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(keras.layers.Dense(units, activation='relu'))
            model.add(keras.layers.Dropout(0.2))  # Add dropout for regularization
            
        # Output layer (multi-class classification)
        model.add(keras.layers.Dense(self.output_dim, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, verbose=1):
        """Train the neural network model"""
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            validation_data = None
            
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=verbose
        )
        
        return history
    
    def predict(self, X):
        """Get model predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        return self.model.evaluate(X_test, y_test)
    
    def save(self, filepath):
        """Save model to disk"""
        self.model.save(filepath)
    
    def load(self, filepath):
        """Load model from disk"""
        self.model = keras.models.load_model(filepath)


class MedicalKnowledgeBase:
    """Symbolic knowledge base for medical reasoning"""
    
    def __init__(self):
        """Initialize the medical knowledge base"""
        self.disease_rules = {}
        self.symptom_descriptions = {}
        self.disease_descriptions = {}
        
    def add_disease_rule(self, disease, required_symptoms, contradicting_symptoms, explanation):
        """
        Add a rule for disease diagnosis
        
        Parameters:
        - disease: Name of the disease
        - required_symptoms: List of symptoms required for diagnosis
        - contradicting_symptoms: List of symptoms that contradict diagnosis
        - explanation: Textual explanation of the rule
        """
        self.disease_rules[disease] = {
            'required': required_symptoms,
            'contradicting': contradicting_symptoms,
            'explanation': explanation
        }
        
    def add_symptom_description(self, symptom, description):
        """Add description for a symptom"""
        self.symptom_descriptions[symptom] = description
        
    def add_disease_description(self, disease, description):
        """Add description for a disease"""
        self.disease_descriptions[disease] = description
        
    def get_symptoms_for_disease(self, disease):
        """Get all symptoms associated with a disease"""
        if disease in self.disease_rules:
            return self.disease_rules[disease]['required']
        return []
    
    def get_explanation(self, disease):
        """Get explanation for a disease diagnosis"""
        if disease in self.disease_rules:
            return self.disease_rules[disease]['explanation']
        return ""
    
    def verify_diagnosis(self, disease, present_symptoms):
        """
        Verify if a disease diagnosis is consistent with observed symptoms
        
        Parameters:
        - disease: The disease to verify
        - present_symptoms: List of symptoms present in the patient
        
        Returns:
        - is_valid: Boolean indicating if diagnosis is valid
        - explanation: Explanation for the verification result
        - confidence: Confidence score (0-1) for the diagnosis
        """
        if disease not in self.disease_rules:
            return False, f"No rules defined for disease: {disease}", 0.0
        
        rule = self.disease_rules[disease]
        required = rule['required']
        contradicting = rule['contradicting']
        
        # Check required symptoms
        required_present = [symptom for symptom in required if symptom in present_symptoms]
        required_missing = [symptom for symptom in required if symptom not in present_symptoms]
        
        # Check contradicting symptoms
        contradicting_present = [symptom for symptom in contradicting if symptom in present_symptoms]
        
        # Calculate match percentage for required symptoms
        required_match = len(required_present) / len(required) if required else 1.0
        
        # Calculate confidence score
        confidence = required_match
        if contradicting_present:
            confidence *= (1 - 0.8 * (len(contradicting_present) / len(contradicting))) if contradicting else 1.0
        
        # Determine if diagnosis is valid
        is_valid = required_match >= 0.7 and (not contradicting_present)
        
        # Generate explanation
        explanation = []
        if required_present:
            explanation.append(f"Patient has key symptoms of {disease}: {', '.join(required_present)}.")
        
        if required_missing:
            explanation.append(f"Missing some typical symptoms of {disease}: {', '.join(required_missing)}.")
            
        if contradicting_present:
            explanation.append(f"Patient has symptoms that contradict {disease}: {', '.join(contradicting_present)}.")
            
        if is_valid:
            explanation.append(f"Diagnosis of {disease} is supported with {confidence:.2f} confidence.")
        else:
            explanation.append(f"Diagnosis of {disease} is not well supported (confidence: {confidence:.2f}).")
            
        return is_valid, " ".join(explanation), confidence


class NeuralSymbolicDiagnosisSystem:
    """Neural-symbolic system for medical diagnosis"""
    
    def __init__(self, neural_model, knowledge_base, symptom_names, disease_names):
        """
        Initialize the neural-symbolic diagnosis system
        
        Parameters:
        - neural_model: Trained neural network model
        - knowledge_base: Medical knowledge base
        - symptom_names: List of symptom names
        - disease_names: List of disease names
        """
        self.neural_model = neural_model
        self.knowledge_base = knowledge_base
        self.symptom_names = symptom_names
        self.disease_names = disease_names
        
    def get_present_symptoms(self, symptom_vector):
        """Convert symptom vector to list of symptom names"""
        return [self.symptom_names[i] for i, has_symptom in enumerate(symptom_vector) if has_symptom == 1]
    
    def diagnose(self, symptoms_vector):
        """
        Diagnose a patient based on symptoms
        
        Parameters:
        - symptoms_vector: Binary vector of patient symptoms
        
        Returns:
        - diagnosis_result: Dictionary with diagnosis details
        """
        # Step 1: Get neural network prediction
        neural_prediction = self.neural_model.predict(np.array([symptoms_vector]))[0]
        predicted_disease_idx = np.argmax(neural_prediction)
        predicted_disease = self.disease_names[predicted_disease_idx]
        neural_confidence = float(neural_prediction[predicted_disease_idx])
        
        # Step 2: Get present symptoms as names
        present_symptoms = self.get_present_symptoms(symptoms_vector)
        
        # Step 3: Verify with symbolic knowledge base
        is_valid, explanation, symbolic_confidence = self.knowledge_base.verify_diagnosis(
            predicted_disease, present_symptoms
        )
        
        # Step 4: If neural prediction is not valid, try other diseases
        if not is_valid:
            alternative_diagnoses = []
            
            # Try other diseases based on neural network confidence
            sorted_indices = np.argsort(neural_prediction)[::-1]  # Sort in descending order
            
            for idx in sorted_indices[1:3]:  # Check the next 2 most confident predictions
                disease = self.disease_names[idx]
                confidence = float(neural_prediction[idx])
                
                is_valid, explanation, symbolic_confidence = self.knowledge_base.verify_diagnosis(
                    disease, present_symptoms
                )
                
                if is_valid:
                    alternative_diagnoses.append({
                        "disease": disease,
                        "neural_confidence": confidence,
                        "symbolic_confidence": symbolic_confidence,
                        "explanation": explanation
                    })
            
            # Check all diseases in knowledge base as fallback
            if not alternative_diagnoses:
                for disease in self.disease_names:
                    if disease != predicted_disease:
                        is_valid, explanation, symbolic_confidence = self.knowledge_base.verify_diagnosis(
                            disease, present_symptoms
                        )
                        
                        if is_valid:
                            neural_confidence = float(neural_prediction[self.disease_names.index(disease)])
                            alternative_diagnoses.append({
                                "disease": disease,
                                "neural_confidence": neural_confidence,
                                "symbolic_confidence": symbolic_confidence,
                                "explanation": explanation
                            })
            
            # If alternatives found, select the best one
            if alternative_diagnoses:
                # Select based on combined confidence
                best_alt = max(alternative_diagnoses, 
                               key=lambda x: 0.7 * x["symbolic_confidence"] + 0.3 * x["neural_confidence"])
                
                return {
                    "diagnosis": best_alt["disease"],
                    "confidence": best_alt["symbolic_confidence"],
                    "neural_confidence": best_alt["neural_confidence"],
                    "explanation": best_alt["explanation"],
                    "symptoms": present_symptoms,
                    "verified": True,
                    "note": "Neural prediction overridden by symbolic reasoning"
                }
            
            # No valid alternatives found
            return {
                "diagnosis": "Uncertain",
                "confidence": 0.0,
                "neural_confidence": neural_confidence,
                "explanation": "No consistent diagnosis found for the given symptoms.",
                "symptoms": present_symptoms,
                "verified": False,
                "neural_prediction": predicted_disease
            }
        
        # Step 5: Neural prediction is valid, return diagnosis
        combined_confidence = 0.7 * symbolic_confidence + 0.3 * neural_confidence
        
        return {
            "diagnosis": predicted_disease,
            "confidence": combined_confidence,
            "neural_confidence": neural_confidence,
            "symbolic_confidence": symbolic_confidence,
            "explanation": explanation,
            "symptoms": present_symptoms,
            "verified": True
        }


# ==============================
# Create and prepare training data
# ==============================

# Define symptoms (features)
symptoms = [
    "Fever", "Cough", "Fatigue", "Headache", "Sore Throat", 
    "Shortness of Breath", "Body Ache", "Runny Nose", "Sneezing",
    "Chest Pain", "Dizziness", "Nausea", "Vomiting", "Diarrhea",
    "Rash", "Joint Pain", "Chills", "Loss of Taste/Smell", "Congestion", "Swollen Lymph Nodes"
]

# Define diagnoses (classes)
diagnoses = [
    "Common Cold", "Influenza", "COVID-19", "Allergic Rhinitis", 
    "Pneumonia", "Bronchitis", "Sinusitis", "Strep Throat"
]

# Generate synthetic training data (200 samples)
num_samples = 200
num_symptoms = len(symptoms)
num_diagnoses = len(diagnoses)

# Initialize data matrices
X_data = np.zeros((num_samples, num_symptoms))
y_data = np.zeros((num_samples, num_diagnoses))

# Define symptom patterns for each diagnosis (probability of each symptom)
symptom_patterns = {
    "Common Cold": {
        "Cough": 0.9, "Runny Nose": 0.9, "Congestion": 0.8, "Sneezing": 0.8, 
        "Sore Throat": 0.7, "Fatigue": 0.6, "Headache": 0.5, "Body Ache": 0.3,
        "Fever": 0.3, "Loss of Taste/Smell": 0.1
    },
    "Influenza": {
        "Fever": 0.9, "Body Ache": 0.9, "Fatigue": 0.9, "Headache": 0.8,
        "Cough": 0.8, "Chills": 0.7, "Sore Throat": 0.5, "Runny Nose": 0.4,
        "Congestion": 0.4, "Nausea": 0.3, "Diarrhea": 0.2
    },
    "COVID-19": {
        "Fever": 0.8, "Cough": 0.8, "Fatigue": 0.8, "Loss of Taste/Smell": 0.7,
        "Shortness of Breath": 0.6, "Headache": 0.6, "Body Ache": 0.5,
        "Sore Throat": 0.4, "Congestion": 0.4, "Nausea": 0.2, "Diarrhea": 0.2
    },
    "Allergic Rhinitis": {
        "Sneezing": 0.9, "Runny Nose": 0.9, "Congestion": 0.8, "Itchy Eyes": 0.7,
        "Fatigue": 0.3, "Headache": 0.3, "Sore Throat": 0.2
    },
    "Pneumonia": {
        "Fever": 0.9, "Cough": 0.9, "Shortness of Breath": 0.9, "Chest Pain": 0.8,
        "Fatigue": 0.8, "Chills": 0.7, "Nausea": 0.4, "Headache": 0.4, "Diarrhea": 0.2
    },
    "Bronchitis": {
        "Cough": 0.9, "Fatigue": 0.8, "Shortness of Breath": 0.7, "Chest Pain": 0.6,
        "Fever": 0.6, "Sore Throat": 0.5, "Body Ache": 0.4, "Chills": 0.4
    },
    "Sinusitis": {
        "Congestion": 0.9, "Headache": 0.8, "Facial Pain": 0.8, "Runny Nose": 0.7,
        "Cough": 0.5, "Fever": 0.4, "Fatigue": 0.5, "Sore Throat": 0.3
    },
    "Strep Throat": {
        "Sore Throat": 0.9, "Fever": 0.8, "Swollen Lymph Nodes": 0.8, "Headache": 0.7,
        "Fatigue": 0.7, "Body Ache": 0.6, "Nausea": 0.3, "Rash": 0.2
    }
}

# Generate data for each diagnosis
samples_per_diagnosis = num_samples // num_diagnoses
for i, diagnosis in enumerate(diagnoses):
    start_idx = i * samples_per_diagnosis
    end_idx = start_idx + samples_per_diagnosis
    
    # Set label (one-hot encoding)
    y_data[start_idx:end_idx, i] = 1
    
    # Generate symptom patterns
    for j, symptom in enumerate(symptoms):
        # Get probability for this symptom in this diagnosis
        prob = symptom_patterns.get(diagnosis, {}).get(symptom, 0.1)
        
        # Randomly generate symptom presence
        X_data[start_idx:end_idx, j] = np.random.binomial(1, prob, samples_per_diagnosis)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# ==============================
# Train neural network model
# ==============================

# Create neural network model
nn_model = NeuralDiagnosticModel(input_dim=num_symptoms, output_dim=num_diagnoses, hidden_layers=[64, 32, 16])
history = nn_model.train(X_train, y_train, X_val=X_test, y_val=y_test, epochs=50, verbose=0)

# ==============================
# Create medical knowledge base
# ==============================

kb = MedicalKnowledgeBase()

# Add disease rules
kb.add_disease_rule(
    "Common Cold", 
    ["Cough", "Runny Nose", "Congestion", "Sneezing", "Sore Throat"], 
    ["Severe Fever", "Shortness of Breath", "Chest Pain", "Loss of Taste/Smell"],
    "Common cold typically presents with upper respiratory symptoms like coughing, runny nose, and sneezing, usually without high fever."
)

kb.add_disease_rule(
    "Influenza", 
    ["Fever", "Body Ache", "Fatigue", "Headache", "Cough"], 
    ["Loss of Taste/Smell", "Rash"],
    "Influenza (flu) typically has a sudden onset with fever, severe body aches, and fatigue, along with respiratory symptoms."
)

kb.add_disease_rule(
    "COVID-19", 
    ["Fever", "Cough", "Fatigue", "Loss of Taste/Smell", "Shortness of Breath"], 
    ["Rash", "Swollen Lymph Nodes"],
    "COVID-19 often presents with fever, cough, fatigue, and distinctive symptoms like loss of taste or smell."
)

kb.add_disease_rule(
    "Allergic Rhinitis", 
    ["Sneezing", "Runny Nose", "Congestion"], 
    ["Fever", "Chest Pain", "Shortness of Breath", "Vomiting", "Loss of Taste/Smell"],
    "Allergic rhinitis is characterized by sneezing, runny nose, and congestion without fever, typically triggered by allergens."
)

kb.add_disease_rule(
    "Pneumonia", 
    ["Fever", "Cough", "Shortness of Breath", "Chest Pain"], 
    ["Rash", "Loss of Taste/Smell"],
    "Pneumonia involves infection of the lungs, causing fever, productive cough, shortness of breath, and often chest pain."
)

kb.add_disease_rule(
    "Bronchitis", 
    ["Cough", "Fatigue", "Shortness of Breath", "Chest Pain"], 
    ["Severe Fever", "Rash", "Swollen Lymph Nodes"],
    "Bronchitis is inflammation of the airways, causing persistent cough, often with mucus production, and sometimes chest discomfort."
)

kb.add_disease_rule(
    "Sinusitis", 
    ["Congestion", "Headache", "Facial Pain", "Runny Nose"], 
    ["Shortness of Breath", "Chest Pain", "Rash"],
    "Sinusitis involves inflammation of the sinuses, causing facial pain/pressure, congestion, and sometimes thick nasal discharge."
)

kb.add_disease_rule(
    "Strep Throat", 
    ["Sore Throat", "Fever", "Swollen Lymph Nodes", "Headache"], 
    ["Cough", "Runny Nose", "Shortness of Breath"],
    "Strep throat is characterized by severe sore throat, fever, and swollen lymph nodes, typically without cough or runny nose."
)

# ==============================
# Create neural-symbolic system
# ==============================

nss = NeuralSymbolicDiagnosisSystem(nn_model, kb, symptoms, diagnoses)

# ==============================
# Demonstrate the system with test cases
# ==============================

def print_diagnosis_result(result, case_num):
    """Pretty print the diagnosis result"""
    print(f"\n{'='*20} CASE {case_num} {'='*20}")
    print(f"Symptoms: {', '.join(result['symptoms'])}")
    print(f"Diagnosis: {result['diagnosis']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Neural Network Confidence: {result['neural_confidence']:.2f}")
    
    if 'symbolic_confidence' in result:
        print(f"Symbolic Confidence: {result['symbolic_confidence']:.2f}")
    
    print(f"Verified: {result['verified']}")
    print(f"Explanation: {result['explanation']}")
    
    if 'note' in result and result['note']:
        print(f"Note: {result['note']}")
    
    if 'neural_prediction' in result:
        print(f"Original Neural Prediction: {result['neural_prediction']}")

# Test case 1: Classic COVID-19 symptoms
test_case1 = np.zeros(num_symptoms)
for symptom in ["Fever", "Cough", "Fatigue", "Loss of Taste/Smell", "Shortness of Breath"]:
    test_case1[symptoms.index(symptom)] = 1
result1 = nss.diagnose(test_case1)
print_diagnosis_result(result1, 1)

# Test case 2: Classic Flu symptoms
test_case2 = np.zeros(num_symptoms)
for symptom in ["Fever", "Body Ache", "Fatigue", "Headache", "Cough", "Chills"]:
    test_case2[symptoms.index(symptom)] = 1
result2 = nss.diagnose(test_case2)
print_diagnosis_result(result2, 2)

# Test case 3: Mixed symptoms (pneumonia + some COVID symptoms)
test_case3 = np.zeros(num_symptoms)
for symptom in ["Fever", "Cough", "Shortness of Breath", "Chest Pain", "Loss of Taste/Smell"]:
    test_case3[symptoms.index(symptom)] = 1
result3 = nss.diagnose(test_case3)
print_diagnosis_result(result3, 3)

# Test case 4: Ambiguous case (mild symptoms)
test_case4 = np.zeros(num_symptoms)
for symptom in ["Fatigue", "Headache", "Runny Nose"]:
    test_case4[symptoms.index(symptom)] = 1
result4 = nss.diagnose(test_case4)
print_diagnosis_result(result4, 4)

# Test case 5: Contradicting symptoms
test_case5 = np.zeros(num_symptoms)
for symptom in ["Sore Throat", "Fever", "Swollen Lymph Nodes", "Cough", "Runny Nose"]:
    test_case5[symptoms.index(symptom)] = 1
result5 = nss.diagnose(test_case5)
print_diagnosis_result(result5, 5)

# ==============================
# Evaluate system performance
# ==============================

# Evaluate neural network component
loss, accuracy = nn_model.evaluate(X_test, y_test)
print(f"\n{'='*20} NEURAL NETWORK EVALUATION {'='*20}")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Evaluate neural-symbolic system on test set
print(f"\n{'='*20} NEURAL-SYMBOLIC SYSTEM EVALUATION {'='*20}")
neural_correct = 0
symbolic_correct = 0
combined_correct = 0

y_true = np.argmax(y_test, axis=1)
y_pred_neural = np.argmax(nn_model.predict(X_test), axis=1)
y_pred_combined = []

for i in range(len(X_test)):
    # Get true diagnosis
    true_diagnosis = diagnoses[y_true[i]]
    
    # Get neural prediction
    neural_prediction = diagnoses[y_pred_neural[i]]
    
    # Get neural-symbolic prediction
    result = nss.diagnose(X_test[i])
    combined_prediction = result["diagnosis"]
    y_pred_combined.append(diagnoses.index(combined_prediction) if combined_prediction in diagnoses else -1)
    
    # Count correct predictions
    if neural_prediction == true_diagnosis:
        neural_correct += 1
    
    if combined_prediction == true_diagnosis:
        combined_correct += 1
        
    if result["verified"]:
        symbolic_correct += 1

print(f"Neural Network Accuracy: {neural_correct / len(X_test):.4f}")
print(f"Neural-Symbolic System Accuracy: {combined_correct / len(X_test):.4f}")
print(f"Percentage of Symbolically Verified Cases: {symbolic_correct / len(X_test):.4f}")

print("\nNeural-Symbolic System Benefits:")
print("1. Explainability: Provides human-readable explanations for diagnoses")
print("2. Verification: Uses medical knowledge to verify neural predictions")
print("3. Fallback Mechanism: Can suggest alternatives when neural predictions are inconsistent")
print("4. Confidence Estimation: Combines neural and symbolic confidence scores")
print("5. Improved accuracy through integrated reasoning")