import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score

class EnhancedNeuralDiagnoser(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # Create dynamic layers with dropout
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class EnhancedNEXUSModel:
    def __init__(self, neural_model, feature_names, target_names, existing_kg=None):
        self.neural = neural_model
        self.feature_names = feature_names
        self.target_names = target_names
        self.kg = existing_kg if existing_kg else KnowledgeGraphEnhancer()
        self.softmax = nn.Softmax(dim=1)
        
        # Clinical decision thresholds (Z-scores)
        self.thresholds = {
            'diabetes': ('glucose_level', 2.0),    # > 200 mg/dL
            'hypertension': ('systolic_bp', 2.5),  # > 140 mmHg
            'asthma': ('wheezing_severity', 1.8)   # Severe wheezing
        }

    def predict(self, X):
        # Neural predictions
        with torch.no_grad():
            neural_probs = self.softmax(self.neural(torch.FloatTensor(X))).numpy()
        neural_preds = np.argmax(neural_probs, axis=1)
        neural_conf = np.max(neural_probs, axis=1)

        # Symbolic reasoning
        symbolic_preds = []
        explanations = []
        for i, sample in enumerate(X):
            final_pred = neural_preds[i]
            explanation = "Neural prediction"
            
            # Check clinical thresholds
            for disease, (feature, threshold) in self.thresholds.items():
                if feature in self.feature_names:
                    feat_idx = self.feature_names.index(feature)
                    if sample[feat_idx] > threshold:
                        disease_idx = self.target_names.index(disease)
                        if disease_idx != neural_preds[i]:
                            final_pred = disease_idx
                            explanation = f"Symbolic override: {feature} > {threshold}"
                            break
                            
            symbolic_preds.append(final_pred)
            explanations.append(explanation)

        return (symbolic_preds, explanations, neural_preds, 
                symbolic_preds, neural_conf, np.ones(len(X)))

    def learn_from_feedback(self, X, y, preds):
        updates = 0
        for i in range(len(X)):
            true_label = self.target_names[y[i]]
            pred_label = self.target_names[preds[i]]
            
            if pred_label != true_label:
                # Find important features
                for feat in self.thresholds.values():
                    feature_name = feat[0]
                    if feature_name in self.feature_names:
                        idx = self.feature_names.index(feature_name)
                        if abs(X[i][idx]) > 1.5:
                            # Strengthen relationship
                            self.kg.add_edge(feature_name, true_label, 
                                           'corrective_relationship', 
                                           weight=0.7)
                            updates += 1
        return updates

    def get_kg(self):
        return self.kg