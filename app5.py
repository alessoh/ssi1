import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

# ======================
# 1. Synthetic Dataset
# ======================
class MedicalDataset:
    def __init__(self, n_samples=1000):
        self.symptoms = ['fever', 'cough', 'fatigue', 'shortness_of_breath']
        self.rules = {
            'severe_flu': lambda x: (x[0] > 101) and (x[1] > 7),
            'pneumonia': lambda x: (x[3] > 8) and (x[0] > 100)
        }
        
        # Generate synthetic data
        self.data = np.random.rand(n_samples, 4) * [5, 10, 10, 10] + [97, 0, 0, 0]
        self.labels = self._generate_labels()
    
    def _generate_labels(self):
        labels = []
        for x in self.data:
            if self.rules['severe_flu'](x):
                labels.append(2)
            elif self.rules['pneumonia'](x):
                labels.append(1)
            else:
                labels.append(0)
        return np.array(labels)

# ======================
# 2. Neural Component (Fixed)
# ======================
class NeuralDiagnoser(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )  # Parenthesis added
    
    def forward(self, x):
        return self.layers(x)

# ======================
# 3. Symbolic Component
# ======================
class SymbolicEngine:
    def __init__(self, rules):
        self.rules = rules
        
    def predict(self, x):
        if self.rules['severe_flu'](x):
            return 2, 0.9
        elif self.rules['pneumonia'](x):
            return 1, 0.85
        return 0, 0.7

# ======================
# 4. NEXUS Integrator
# ======================
class NexusModel:
    def __init__(self, neural_model, symbolic_engine):
        self.neural = neural_model
        self.symbolic = symbolic_engine
        self.softmax = nn.Softmax(dim=1)
        
    def predict(self, x):
        neural_out = self.neural(torch.FloatTensor(x))
        neural_probs = self.softmax(neural_out).detach().numpy()
        neural_conf = np.max(neural_probs, axis=1)
        
        symbolic_preds = []
        for case in x:
            pred, _ = self.symbolic.predict(case)
            symbolic_preds.append(pred)
        
        combined = []
        explanations = []
        for n_conf, n_prob, s_pred in zip(neural_conf, neural_probs, symbolic_preds):
            if n_conf > 0.8:
                final_pred = np.argmax(n_prob)
                exp = "Neural dominant"
            else:
                final_pred = s_pred
                exp = "Symbolic dominant"
            combined.append(final_pred)
            explanations.append(exp)
            
        return combined, explanations

# ======================
# 5. Training & Evaluation (Fixed)
# ======================
dataset = MedicalDataset(1000)
neural_net = NeuralDiagnoser()
symbolic_engine = SymbolicEngine(dataset.rules)
nexus = NexusModel(neural_net, symbolic_engine)

# Train
X = torch.FloatTensor(dataset.data)
y = torch.LongTensor(dataset.labels)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(neural_net.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = neural_net(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# Evaluate on TEST data
test_set = MedicalDataset(200)
nexus_preds, explanations = nexus.predict(test_set.data)

# Corrected evaluation
test_X = torch.FloatTensor(test_set.data)
neural_test_outputs = neural_net(test_X)
neural_test_preds = np.argmax(neural_test_outputs.detach().numpy(), axis=1)

print(f"Neural Only Accuracy: {accuracy_score(test_set.labels, neural_test_preds):.2f}")
print(f"NEXUS Accuracy: {accuracy_score(test_set.labels, nexus_preds):.2f}")

# Example output
print("\nExample Decisions:")
for i in range(3):
    print(f"Case {i+1}: {explanations[i]}")
    print(f"True: {test_set.labels[i]}, Predicted: {nexus_preds[i]}\n")