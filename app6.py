import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# ======================
# 1. Enhanced MedicalDataset
# ======================
class MedicalDataset:
    def __init__(self, n_samples=1000, imbalance_ratio=(0.70, 0.20, 0.10)):
        self.rules = {
            'severe_flu': lambda x: (x[0] > 101) and (x[1] > 7),
            'pneumonia': lambda x: (x[3] > 7) or (x[2] > 6)
        }
        
        n_mild = int(n_samples * imbalance_ratio[0])
        n_pneumonia = int(n_samples * imbalance_ratio[1])
        n_severe = n_samples - n_mild - n_pneumonia
        
        mild_data = np.random.rand(n_mild, 4) * [3, 5, 5, 5] + [97, 0, 0, 0]
        pneumonia_data = np.random.rand(n_pneumonia, 4) * [3, 5, 5, 5] + [98, 5, 5, 5]
        pneumonia_data[:, 3] = np.random.rand(n_pneumonia) * 5 + 7
        severe_data = np.random.rand(n_severe, 4) * [3, 5, 5, 5] + [101, 7, 5, 5]
        
        self.data = np.vstack([mild_data, pneumonia_data, severe_data])
        self.labels = np.concatenate([
            np.zeros(n_mild),
            np.ones(n_pneumonia),
            np.full(n_severe, 2)
        ]).astype(int)

# ======================
# 2. Neural Component
# ======================
class NeuralDiagnoser(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )
    
    def forward(self, x):
        return self.layers(x)

# ======================
# 3. Enhanced Symbolic Engine
# ======================
class SymbolicEngine:
    def __init__(self, rules):
        self.rules = rules
        
    def predict(self, x):
        if self.rules['severe_flu'](x):
            return 2, 0.95
        elif self.rules['pneumonia'](x):
            pneumonia_conf = min(1.0, 0.7 + 0.05*(x[3]-7) + 0.05*(x[2]-6))
            return 1, round(pneumonia_conf, 2)
        else:
            mild_conf = max(0.5, 0.7 - 0.1*(x[0]/102))
            return 0, round(mild_conf, 2)

# ======================
# 4. Improved Nexus Model
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
        neural_preds = np.argmax(neural_probs, axis=1)
        
        symbolic_preds = []
        symbolic_confs = []
        for case in x:
            pred, conf = self.symbolic.predict(case)
            symbolic_preds.append(pred)
            symbolic_confs.append(conf)
        
        combined = []
        explanations = []
        for n_conf, n_pred, s_pred, s_conf in zip(neural_conf, neural_preds, 
                                                 symbolic_preds, symbolic_confs):
            if n_conf > 0.7 and s_conf < 0.8:
                final_pred = n_pred
                exp = f"Neural ({n_conf:.2f} > 0.7)"
            else:
                final_pred = s_pred
                exp = f"Symbolic ({s_conf:.2f} >= {n_conf:.2f})"
            
            combined.append(final_pred)
            explanations.append(exp)
            
        return combined, explanations

    def predict_with_confidence(self, x):
        neural_out = self.neural(torch.FloatTensor(x))
        neural_probs = self.softmax(neural_out).detach().numpy()
        neural_conf = np.max(neural_probs, axis=1)
        
        symbolic_confs = []
        for case in x:
            _, conf = self.symbolic.predict(case)
            symbolic_confs.append(conf)
            
        return neural_conf, symbolic_confs

# ======================
# 5. Visualization Tools
# ======================
def plot_confidences(neural_confs, symbolic_confs):
    plt.figure(figsize=(10, 4))
    
    plt.subplot(121)
    plt.hist(neural_confs, bins=20, color='blue', alpha=0.7)
    plt.title("Neural Confidence Distribution")
    
    plt.subplot(122)
    plt.hist(symbolic_confs, bins=20, color='green', alpha=0.7)
    plt.title("Symbolic Confidence Distribution")
    
    plt.tight_layout()
    plt.show()

# ======================
# 6. Training & Evaluation
# ======================
if __name__ == "__main__":
    # Initialize components
    dataset = MedicalDataset(2000, imbalance_ratio=(0.70, 0.20, 0.10))
    neural_net = NeuralDiagnoser()
    symbolic_engine = SymbolicEngine(dataset.rules)
    nexus = NexusModel(neural_net, symbolic_engine)
    
    # Training
    X = torch.FloatTensor(dataset.data)
    y = torch.LongTensor(dataset.labels)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=0.005)
    
    # Training loop
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = neural_net(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    # Evaluation
    test_set = MedicalDataset(500, imbalance_ratio=(0.33, 0.33, 0.34))
    nexus_preds, explanations = nexus.predict(test_set.data)
    neural_conf, symbolic_conf = nexus.predict_with_confidence(test_set.data)
    
    # Performance analysis
    print("\n=== Neural Only ===")
    test_outputs = neural_net(torch.FloatTensor(test_set.data))
    neural_preds = np.argmax(test_outputs.detach().numpy(), axis=1)
    print(classification_report(test_set.labels, neural_preds,
                               target_names=['mild', 'pneumonia', 'severe']))
    
    print("\n=== NEXUS ===")
    print(classification_report(test_set.labels, nexus_preds,
                               target_names=['mild', 'pneumonia', 'severe']))
    
    # Confidence visualization
    plot_confidences(neural_conf, symbolic_conf)
    
    # Error analysis
    print("\n=== Critical Corrections ===")
    corrections = []
    for i, (true, pred_n, pred_nx) in enumerate(zip(test_set.labels, neural_preds, nexus_preds)):
        if pred_n != true and pred_nx == true and true in [1, 2]:
            corrections.append(i)
            if len(corrections) >= 5:
                break
                
    for idx in corrections:
        print(f"\nCase {idx+1}:")
        print(f"Symptoms: {test_set.data[idx]}")
        print(f"True: {test_set.labels[idx]} ({['mild', 'pneumonia', 'severe'][test_set.labels[idx]]})")
        print(f"Neural: {neural_preds[idx]} ({['mild', 'pneumonia', 'severe'][neural_preds[idx]]})")
        print(f"NEXUS: {nexus_preds[idx]} ({['mild', 'pneumonia', 'severe'][nexus_preds[idx]]})")
        print(f"Decision: {explanations[idx]}")