import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

def generate_synthetic_patients(n_patients=1000, seed=42):
    """
    Generate synthetic medical data for patients with various conditions.
    
    Returns:
        DataFrame with patient symptoms and ground truth diagnoses
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Define our symptoms and risk factors
    symptoms = ['fever', 'high_fever', 'cough', 'severe_cough', 'fatigue', 
                'headache', 'shortness_of_breath', 'sore_throat']
    
    risk_factors = ['elderly', 'immunocompromised', 'child']
    
    # Define our conditions and their probability distributions
    conditions = {
        'common_cold': 0.40,
        'flu': 0.25,
        'covid': 0.20,
        'pneumonia': 0.10,
        'allergies': 0.05
    }
    
    # Define symptom probabilities given condition
    # Format: {condition: {symptom: probability}}
    symptom_probs = {
        'common_cold': {
            'fever': 0.3, 'high_fever': 0.05, 'cough': 0.8, 'severe_cough': 0.1,
            'fatigue': 0.6, 'headache': 0.5, 'shortness_of_breath': 0.05, 'sore_throat': 0.9
        },
        'flu': {
            'fever': 0.9, 'high_fever': 0.6, 'cough': 0.8, 'severe_cough': 0.3,
            'fatigue': 0.9, 'headache': 0.7, 'shortness_of_breath': 0.3, 'sore_throat': 0.5
        },
        'covid': {
            'fever': 0.8, 'high_fever': 0.4, 'cough': 0.9, 'severe_cough': 0.5,
            'fatigue': 0.8, 'headache': 0.6, 'shortness_of_breath': 0.7, 'sore_throat': 0.4
        },
        'pneumonia': {
            'fever': 0.9, 'high_fever': 0.7, 'cough': 0.8, 'severe_cough': 0.8,
            'fatigue': 0.9, 'headache': 0.3, 'shortness_of_breath': 0.9, 'sore_throat': 0.2
        },
        'allergies': {
            'fever': 0.05, 'high_fever': 0.01, 'cough': 0.6, 'severe_cough': 0.1,
            'fatigue': 0.3, 'headache': 0.4, 'shortness_of_breath': 0.4, 'sore_throat': 0.3
        }
    }
    
    # Define risk factor probabilities in the population
    risk_factor_probs = {
        'elderly': 0.20,
        'immunocompromised': 0.05,
        'child': 0.15
    }
    
    # Define treatment needs based on condition and risk factors
    # Format: (condition, risk_factor, treatment, probability)
    treatment_rules = [
        # Common cold treatments
        ('common_cold', None, 'need_rest', 0.95),
        ('common_cold', None, 'need_testing', 0.10),
        
        # Flu treatments
        ('flu', None, 'need_rest', 0.95),
        ('flu', None, 'need_testing', 0.50),
        ('flu', 'elderly', 'need_hospitalization', 0.30),
        ('flu', 'immunocompromised', 'need_hospitalization', 0.60),
        ('flu', 'child', 'need_hospitalization', 0.20),
        
        # COVID treatments
        ('covid', None, 'need_rest', 0.90),
        ('covid', None, 'need_testing', 0.98),
        ('covid', None, 'need_hospitalization', 0.30),
        ('covid', 'elderly', 'need_hospitalization', 0.70),
        ('covid', 'immunocompromised', 'need_hospitalization', 0.85),
        ('covid', 'child', 'need_hospitalization', 0.25),
        ('covid', 'elderly', 'need_ventilator', 0.30),
        ('covid', 'immunocompromised', 'need_ventilator', 0.50),
        
        # Pneumonia treatments
        ('pneumonia', None, 'need_rest', 0.80),
        ('pneumonia', None, 'need_testing', 0.90),
        ('pneumonia', None, 'need_hospitalization', 0.70),
        ('pneumonia', 'elderly', 'need_hospitalization', 0.95),
        ('pneumonia', 'immunocompromised', 'need_hospitalization', 0.98),
        ('pneumonia', 'child', 'need_hospitalization', 0.80),
        ('pneumonia', None, 'need_ventilator', 0.20),
        ('pneumonia', 'elderly', 'need_ventilator', 0.50),
        ('pneumonia', 'immunocompromised', 'need_ventilator', 0.70),
        ('pneumonia', 'child', 'need_ventilator', 0.30),
        
        # Allergy treatments
        ('allergies', None, 'need_rest', 0.50),
        ('allergies', None, 'need_testing', 0.40)
    ]
    
    # Initialize empty DataFrame
    columns = symptoms + risk_factors + list(conditions.keys()) + [
        'need_rest', 'need_testing', 'need_hospitalization', 'need_ventilator'
    ]
    data = pd.DataFrame(0, index=range(n_patients), columns=columns)
    
    # Generate patient data
    for i in range(n_patients):
        # Assign primary condition
        condition = np.random.choice(list(conditions.keys()), p=list(conditions.values()))
        data.loc[i, condition] = 1
        
        # Assign symptoms based on condition
        for symptom in symptoms:
            prob = symptom_probs[condition][symptom]
            data.loc[i, symptom] = 1 if random.random() < prob else 0
            
        # Force logical consistency
        if data.loc[i, 'high_fever'] == 1:
            data.loc[i, 'fever'] = 1
        if data.loc[i, 'severe_cough'] == 1:
            data.loc[i, 'cough'] = 1
        
        # Assign risk factors
        for risk in risk_factors:
            data.loc[i, risk] = 1 if random.random() < risk_factor_probs[risk] else 0
            
        # Assign treatment needs based on condition and risk factors
        for rule in treatment_rules:
            rule_condition, rule_risk, treatment, prob = rule
            if data.loc[i, rule_condition] == 1:
                # If risk factor specified, check if patient has it
                if rule_risk is None or data.loc[i, rule_risk] == 1:
                    data.loc[i, treatment] = 1 if random.random() < prob else 0
    
    # Create a dictionary to track the patient's symptom vectors for the neural-symbolic model
    patient_vectors = []
    for i in range(n_patients):
        # Get the symptoms and risk factors as a list
        symptoms_vector = [int(data.loc[i, col]) for col in symptoms + risk_factors]
        patient_vectors.append(symptoms_vector)
    
    data['symptom_vector'] = patient_vectors
    
    # Split into training and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=seed)
    
    return train_data, test_data

def print_sample_patients(data, n_samples=5):
    """
    Print characteristics of sample patients from the dataset
    """
    symptoms = ['fever', 'high_fever', 'cough', 'severe_cough', 'fatigue', 
                'headache', 'shortness_of_breath', 'sore_throat']
    risk_factors = ['elderly', 'immunocompromised', 'child']
    conditions = ['common_cold', 'flu', 'covid', 'pneumonia', 'allergies']
    treatments = ['need_rest', 'need_testing', 'need_hospitalization', 'need_ventilator']
    
    print(f"\nSample of {n_samples} patients from dataset:")
    for i in range(min(n_samples, len(data))):
        patient = data.iloc[i]
        
        # Get present symptoms, risk factors, conditions, and treatments
        present_symptoms = [s for s in symptoms if patient[s] == 1]
        present_risks = [r for r in risk_factors if patient[r] == 1]
        present_conditions = [c for c in conditions if patient[c] == 1]
        present_treatments = [t for t in treatments if patient[t] == 1]
        
        print(f"\nPatient {i+1}:")
        print(f"  Symptoms: {', '.join(present_symptoms) if present_symptoms else 'None'}")
        print(f"  Risk Factors: {', '.join(present_risks) if present_risks else 'None'}")
        print(f"  Condition: {', '.join(present_conditions)}")
        print(f"  Treatments: {', '.join(present_treatments) if present_treatments else 'None'}")

def analyze_dataset(data, title="Dataset Analysis"):
    """
    Analyze the generated dataset and print statistics
    """
    # Count conditions
    conditions = ['common_cold', 'flu', 'covid', 'pneumonia', 'allergies']
    condition_counts = {cond: data[cond].sum() for cond in conditions}
    
    # Count risk factors
    risk_factors = ['elderly', 'immunocompromised', 'child']
    risk_counts = {risk: data[risk].sum() for risk in risk_factors}
    
    # Count treatments
    treatments = ['need_rest', 'need_testing', 'need_hospitalization', 'need_ventilator']
    treatment_counts = {treat: data[treat].sum() for treat in treatments}
    
    # Count symptom frequency
    symptoms = ['fever', 'high_fever', 'cough', 'severe_cough', 'fatigue', 
                'headache', 'shortness_of_breath', 'sore_throat']
    symptom_counts = {symp: data[symp].sum() for symp in symptoms}
    
    # Print statistics
    print(f"\n{title} (Total patients: {len(data)})")
    print("\nCondition Distribution:")
    for cond, count in condition_counts.items():
        percentage = (count / len(data)) * 100
        print(f"  {cond}: {count} patients ({percentage:.1f}%)")
    
    print("\nRisk Factor Distribution:")
    for risk, count in risk_counts.items():
        percentage = (count / len(data)) * 100
        print(f"  {risk}: {count} patients ({percentage:.1f}%)")
    
    print("\nTreatment Distribution:")
    for treat, count in treatment_counts.items():
        percentage = (count / len(data)) * 100
        print(f"  {treat}: {count} patients ({percentage:.1f}%)")
    
    print("\nSymptom Distribution:")
    for symp, count in symptom_counts.items():
        percentage = (count / len(data)) * 100
        print(f"  {symp}: {count} patients ({percentage:.1f}%)")
    
    # Analyze high-risk patients
    high_risk = data[(data['elderly'] == 1) | (data['immunocompromised'] == 1)]
    print(f"\nHigh-risk patients (elderly or immunocompromised): {len(high_risk)} ({len(high_risk)/len(data)*100:.1f}%)")
    
    # Analyze severe cases
    severe_cases = data[(data['pneumonia'] == 1) | (data['covid'] == 1)]
    print(f"Severe cases (pneumonia or COVID): {len(severe_cases)} ({len(severe_cases)/len(data)*100:.1f}%)")
    
    # Analyze hospitalization needs
    hospitalization = data[data['need_hospitalization'] == 1]
    print(f"Patients requiring hospitalization: {len(hospitalization)} ({len(hospitalization)/len(data)*100:.1f}%)")
    
    # Analyze ventilator needs
    ventilator = data[data['need_ventilator'] == 1]
    print(f"Patients requiring ventilator: {len(ventilator)} ({len(ventilator)/len(data)*100:.1f}%)")

if __name__ == "__main__":
    # Generate 1000 patients
    train_data, test_data = generate_synthetic_patients(1000)
    
    # Print basic information
    print(f"Generated {len(train_data)} training patients and {len(test_data)} test patients")
    
    # Print sample patients
    print_sample_patients(train_data, n_samples=5)
    
    # Analyze datasets
    analyze_dataset(train_data, "Training Dataset Analysis")
    analyze_dataset(test_data, "Test Dataset Analysis")
    
    print("\nSynthetic data generation complete!")