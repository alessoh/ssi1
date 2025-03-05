# Add this code at the appropriate location in your main script
# to fix the test cases section that was cut off

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

# Create the neural-symbolic system
nss = NeuralSymbolicDiagnosisSystem(nn_model, kb, symptoms, diagnoses)

# Test case 1: Classic COVID-19 symptoms
test_case1 = np.zeros(len(symptoms))
for symptom in ["Fever", "Cough", "Fatigue", "Loss of Taste/Smell", "Shortness of Breath"]:
    if symptom in symptoms:
        test_case1[symptoms.index(symptom)] = 1
result1 = nss.diagnose(test_case1)
print_diagnosis_result(result1, 1)

# Test case 2: Classic Flu symptoms
test_case2 = np.zeros(len(symptoms))
for symptom in ["Fever", "Body Ache", "Fatigue", "Headache", "Cough", "Chills"]:
    if symptom in symptoms:
        test_case2[symptoms.index(symptom)] = 1
result2 = nss.diagnose(test_case2)
print_diagnosis_result(result2, 2)

# Test case 3: Mixed symptoms (pneumonia + some COVID symptoms)
test_case3 = np.zeros(len(symptoms))
for symptom in ["Fever", "Cough", "Shortness of Breath", "Chest Pain", "Loss of Taste/Smell"]:
    if symptom in symptoms:
        test_case3[symptoms.index(symptom)] = 1
result3 = nss.diagnose(test_case3)
print_diagnosis_result(result3, 3)

# Test case 4: Ambiguous case (mild symptoms)
test_case4 = np.zeros(len(symptoms))
for symptom in ["Fatigue", "Headache", "Runny Nose"]:
    if symptom in symptoms:
        test_case4[symptoms.index(symptom)] = 1
result4 = nss.diagnose(test_case4)
print_diagnosis_result(result4, 4)

# Test case 5: Contradicting symptoms
test_case5 = np.zeros(len(symptoms))
for symptom in ["Sore Throat", "Fever", "Swollen Lymph Nodes", "Cough", "Runny Nose"]:
    if symptom in symptoms:
        test_case5[symptoms.index(symptom)] = 1
result5 = nss.diagnose(test_case5)
print_diagnosis_result(result5, 5)

# Load patient from file for testing
try:
    print("\n=== Testing with a real patient case from file ===")
    # Get a random patient from the test set
    patient_idx = np.random.randint(0, len(X_test))
    patient_symptoms = X_test[patient_idx]
    actual_diagnosis = diagnoses[np.argmax(y_test[patient_idx])]
    
    # Diagnose the patient
    result = nss.diagnose(patient_symptoms)
    
    # Print results
    print(f"\n{'='*20} PATIENT FROM FILE {'='*20}")
    print(f"Symptoms: {', '.join(result['symptoms'])}")
    print(f"Actual Diagnosis: {actual_diagnosis}")
    print(f"System's Diagnosis: {result['diagnosis']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Verified: {result['verified']}")
    print(f"Explanation: {result['explanation']}")
    
except Exception as e:
    print(f"Could not test with patient from file: {e}")

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