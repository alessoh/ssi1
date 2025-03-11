import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tabulate import tabulate
import random
from torch.utils.data import SubsetRandomSampler

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# ===========================
# 1. Define Neural Component
# ===========================
class SimpleTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ff_network = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, attention_mask=None):
        residual = x
        x = self.layer_norm1(x)
        attn_output, attn_weights = self.self_attn(x, x, x, attn_mask=attention_mask)
        x = residual + attn_output
        residual = x
        x = self.layer_norm2(x)
        ff_output = self.ff_network(x)
        x = residual + ff_output
        return x, attn_weights

class NeuralModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer_layers = nn.ModuleList([
            SimpleTransformerLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        attentions = []
        for layer in self.transformer_layers:
            x, attn = layer(x)
            attentions.append(attn)
        x = x.squeeze(1)
        logits = self.classifier(x)
        return logits, x, attentions

# ===========================
# 2. Define Improved Symbolic Component
# ===========================
class KnowledgeGraph:
    def __init__(self):
        self.entities = {}
        self.relations = []
        self.rules = []
        
    def add_entity(self, entity_id, name):
        self.entities[entity_id] = name
        return self
        
    def add_relation(self, source_id, relation_type, target_id, weight=1.0):
        self.relations.append((source_id, relation_type, target_id, weight))
        return self
        
    def add_rule(self, premise_ids, conclusion_id, confidence=1.0):
        self.rules.append((premise_ids, conclusion_id, confidence))
        return self
    
    def reason(self, active_entities):
        inferred = set(active_entities)
        reasoning_steps = {}
        confidences = {}
        class_scores = {i: 0.0 for i in range(4)}  # 4 classes: Common Cold (0), Flu (1), COVID (2), Pneumonia (3)
        
        # Initialize with given symptoms
        for entity_id in active_entities:
            if entity_id in self.entities:
                reasoning_steps[entity_id] = f"Given: {self.entities[entity_id]}"
                confidences[entity_id] = 1.0
        
        # Apply relations
        for source_id, relation_type, target_id, weight in self.relations:
            if source_id in inferred and target_id not in inferred:
                inferred.add(target_id)
                step = f"{self.entities[source_id]} --{relation_type}--> {self.entities[target_id]}"
                reasoning_steps[target_id] = step
                confidences[target_id] = weight * confidences.get(source_id, 1.0)
        
        # Apply rules and aggregate class scores
        for premise_ids, conclusion_id, confidence in self.rules:
            if all(p_id in inferred for p_id in premise_ids):
                inferred.add(conclusion_id)
                premises = [self.entities[p_id] for p_id in premise_ids]
                step = f"Rule: IF {' AND '.join(premises)} THEN {self.entities[conclusion_id]}"
                reasoning_steps[conclusion_id] = step
                premise_conf = min([confidences.get(p_id, 1.0) for p_id in premise_ids])  # Min confidence of premises
                confidences[conclusion_id] = confidence * premise_conf
                if conclusion_id < 4:  # If conclusion is a class
                    class_scores[conclusion_id] = max(class_scores[conclusion_id], confidences[conclusion_id])
        
        return inferred, reasoning_steps, confidences, class_scores

# ===========================
# 3. Neural-Symbolic Interface
# ===========================
class NeuralSymbolicInterface:
    def __init__(self, input_dim, num_symbols, num_classes):
        self.neural_to_symbol_matrix = torch.randn(num_symbols, input_dim)
        self.symbol_to_class_matrix = torch.zeros(num_symbols, num_classes)
        self.threshold = 0.5
        
    def set_symbol_to_class_mapping(self, symbol_to_class_dict):
        for symbol_id, class_weights in symbol_to_class_dict.items():
            for class_id, weight in class_weights.items():
                self.symbol_to_class_matrix[symbol_id, class_id] = weight
    
    def neural_to_symbolic(self, neural_repr):
        neural_repr_norm = F.normalize(neural_repr, dim=1)
        symbol_embeds_norm = F.normalize(self.neural_to_symbol_matrix, dim=1)
        similarity = torch.mm(neural_repr_norm, symbol_embeds_norm.t())
        activations = (similarity > self.threshold).float()
        return activations, similarity
    
    def symbolic_to_neural_prediction(self, symbolic_activations, confidences=None):
        if confidences is None:
            class_scores = torch.mm(symbolic_activations, self.symbol_to_class_matrix)
        else:
            conf_tensor = torch.zeros_like(symbolic_activations)
            for i, confs in enumerate(confidences):
                for symbol_id, conf in confs.items():
                    conf_tensor[i, symbol_id] = conf
            weighted_activations = symbolic_activations * conf_tensor
            class_scores = torch.mm(weighted_activations, self.symbol_to_class_matrix)
        return class_scores

# ===========================
# 4. Metacognitive Control
# ===========================
class MetacognitiveController:
    def __init__(self, neural_threshold=0.9, symbolic_threshold=0.7):
        self.neural_threshold = neural_threshold
        self.symbolic_threshold = symbolic_threshold
    
    def decide_strategy(self, neural_conf, symbolic_conf, risk_level='medium'):
        neural_threshold = self.neural_threshold
        symbolic_threshold = self.symbolic_threshold
        
        if risk_level == 'high':
            neural_threshold += 0.1
            symbolic_threshold -= 0.1
        elif risk_level == 'low':
            neural_threshold -= 0.1
            symbolic_threshold += 0.1
            
        if neural_conf >= neural_threshold and symbolic_conf < symbolic_threshold:
            return {
                'strategy': 'neural',
                'neural_weight': 1.0,
                'symbolic_weight': 0.0,
                'explanation': f'Using neural prediction (confidence: {neural_conf:.2f})'
            }
        elif symbolic_conf >= symbolic_threshold and neural_conf < neural_threshold:
            return {
                'strategy': 'symbolic',
                'neural_weight': 0.0,
                'symbolic_weight': 1.0,
                'explanation': f'Using symbolic reasoning (confidence: {symbolic_conf:.2f})'
            }
        else:
            total_conf = neural_conf + symbolic_conf
            neural_weight = neural_conf / total_conf if total_conf > 0 else 0.5
            symbolic_weight = 1 - neural_weight
            return {
                'strategy': 'hybrid',
                'neural_weight': neural_weight,
                'symbolic_weight': symbolic_weight,
                'explanation': f'Using weighted combination (neural: {neural_weight:.2f}, symbolic: {symbolic_weight:.2f})'
            }

# ===========================
# 5. NEXUS Integrated Model
# ===========================
class NEXUSModel:
    def __init__(self, input_dim, num_classes, num_symbols, symbol_names, class_names):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_symbols = num_symbols
        self.symbol_names = symbol_names
        self.class_names = class_names
        self.symbol_to_id = {name: i for i, name in enumerate(symbol_names)}
        
        self.neural_model = NeuralModel(input_dim, num_classes)
        self.knowledge_graph = KnowledgeGraph()
        self.interface = NeuralSymbolicInterface(input_dim=128, num_symbols=num_symbols, num_classes=num_classes)
        self.metacognitive = MetacognitiveController()
        
        self.eval_results = {
            'neural': {'correct': 0, 'total': 0, 'confusion': None, 'predictions': [], 'true_labels': []},
            'symbolic': {'correct': 0, 'total': 0, 'confusion': None, 'predictions': [], 'true_labels': []},
            'nexus': {'correct': 0, 'total': 0, 'confusion': None, 'predictions': [], 'true_labels': []}
        }
        
    def init_knowledge_graph(self):
        kg = self.knowledge_graph
        for i, name in enumerate(self.symbol_names):
            kg.add_entity(i, name)
        for i, name in enumerate(self.class_names):
            kg.add_entity(i, name)
        return kg
    
    def train_neural(self, dataloader, num_epochs=10, lr=0.001):
        optimizer = torch.optim.Adam(self.neural_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs, _, _ = self.neural_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_loss += loss.item()
            
            epoch_loss = total_loss / len(dataloader)
            epoch_acc = 100 * correct / total
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    def diagnose(self, x, active_symptoms=None, risk_level='medium'):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            neural_logits, neural_repr, attentions = self.neural_model(x)
            neural_probs = F.softmax(neural_logits, dim=1)
            neural_pred = torch.argmax(neural_probs, dim=1).item()
            neural_conf = neural_probs[0, neural_pred].item()
        
        symbolic_activations, similarities = self.interface.neural_to_symbolic(neural_repr)
        
        if active_symptoms is not None:
            symptom_ids = [self.symbol_to_id[name] for name in active_symptoms if name in self.symbol_to_id]
        else:
            symptom_ids = torch.nonzero(symbolic_activations[0]).squeeze(-1).tolist()
            if not isinstance(symptom_ids, list):
                symptom_ids = [symptom_ids]
                
        inferred_ids, reasoning_steps, confidences, class_scores = self.knowledge_graph.reason(symptom_ids)
        
        symbolic_scores = torch.tensor(list(class_scores.values()), dtype=torch.float32).unsqueeze(0)
        symbolic_probs = F.softmax(symbolic_scores, dim=1)
        symbolic_pred = torch.argmax(symbolic_probs, dim=1).item()
        symbolic_conf = symbolic_probs[0, symbolic_pred].item()
        
        strategy = self.metacognitive.decide_strategy(neural_conf, symbolic_conf, risk_level)
        
        if strategy['strategy'] == 'neural':
            final_pred = neural_pred
            final_conf = neural_conf
        elif strategy['strategy'] == 'symbolic':
            final_pred = symbolic_pred
            final_conf = symbolic_conf
        else:
            combined_probs = (strategy['neural_weight'] * neural_probs + strategy['symbolic_weight'] * symbolic_probs)
            final_pred = torch.argmax(combined_probs, dim=1).item()
            final_conf = combined_probs[0, final_pred].item()
        
        result = {
            'neural': {'prediction': neural_pred, 'confidence': neural_conf, 'class_name': self.class_names[neural_pred]},
            'symbolic': {
                'prediction': symbolic_pred,
                'confidence': symbolic_conf,
                'class_name': self.class_names[symbolic_pred],
                'reasoning_steps': reasoning_steps,
                'inferred_symbols': [self.symbol_names[i] for i in inferred_ids if i < len(self.symbol_names)]
            },
            'nexus': {'prediction': final_pred, 'confidence': final_conf, 'class_name': self.class_names[final_pred], 'strategy': strategy}
        }
        return result
    
    def evaluate(self, dataloader, symptom_dict=None):
        self.eval_results['neural']['confusion'] = np.zeros((self.num_classes, self.num_classes))
        self.eval_results['symbolic']['confusion'] = np.zeros((self.num_classes, self.num_classes))
        self.eval_results['nexus']['confusion'] = np.zeros((self.num_classes, self.num_classes))
        
        for key in self.eval_results:
            self.eval_results[key]['correct'] = 0
            self.eval_results[key]['total'] = 0
        
        agreement_cases = {'all_correct': 0, 'all_wrong': 0, 'neural_only': 0, 'symbolic_only': 0, 'nexus_better': 0}
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                active_symptoms = symptom_dict.get(i, None) if symptom_dict else None
                result = self.diagnose(inputs, active_symptoms)
                true_label = labels.item()
                
                for key in ['neural', 'symbolic', 'nexus']:
                    pred = result[key]['prediction']
                    self.eval_results[key]['confusion'][true_label, pred] += 1
                    self.eval_results[key]['predictions'].append(pred)
                    self.eval_results[key]['true_labels'].append(true_label)
                    if pred == true_label:
                        self.eval_results[key]['correct'] += 1
                    self.eval_results[key]['total'] += 1
                
                neural_correct = result['neural']['prediction'] == true_label
                symbolic_correct = result['symbolic']['prediction'] == true_label
                nexus_correct = result['nexus']['prediction'] == true_label
                
                if neural_correct and symbolic_correct and nexus_correct:
                    agreement_cases['all_correct'] += 1
                elif not neural_correct and not symbolic_correct and not nexus_correct:
                    agreement_cases['all_wrong'] += 1
                elif neural_correct and not symbolic_correct:
                    agreement_cases['neural_only'] += 1
                elif not neural_correct and symbolic_correct:
                    agreement_cases['symbolic_only'] += 1
                elif nexus_correct and (not neural_correct or not symbolic_correct):
                    agreement_cases['nexus_better'] += 1
        
        for key in self.eval_results:
            self.eval_results[key]['accuracy'] = (self.eval_results[key]['correct'] / self.eval_results[key]['total']) if self.eval_results[key]['total'] > 0 else 0
        
        self.eval_results['agreement_cases'] = agreement_cases
        return self.eval_results
    
    def explain_diagnosis(self, result, detail_level='medium'):
        explanation = [f"Diagnosis: {result['nexus']['class_name']} (Confidence: {result['nexus']['confidence']:.2f})"]
        explanation.append(f"Strategy: {result['nexus']['strategy']['strategy']}")
        explanation.append(f"Reason: {result['nexus']['strategy']['explanation']}")
        
        if detail_level in ['medium', 'high']:
            explanation.append("\nSymbolic Reasoning:")
            explanation.append(f"Identified symbols: {', '.join(result['symbolic']['inferred_symbols'])}")
            if detail_level == 'high' and result['symbolic']['reasoning_steps']:
                explanation.append("\nReasoning steps:")
                for symbol_id, step in result['symbolic']['reasoning_steps'].items():
                    if symbol_id < len(self.symbol_names) + len(self.class_names):
                        symbol_name = self.symbol_names[symbol_id] if symbol_id < len(self.symbol_names) else self.class_names[symbol_id - len(self.symbol_names)]
                        explanation.append(f"- {symbol_name}: {step}")
            explanation.append(f"\nNeural model prediction: {result['neural']['class_name']} (Confidence: {result['neural']['confidence']:.2f})")
            explanation.append(f"Symbolic model prediction: {result['symbolic']['class_name']} (Confidence: {result['symbolic']['confidence']:.2f})")
        
        return "\n".join(explanation)
    
    def visualize_results(self):
        if self.eval_results['neural']['confusion'] is None:
            print("No evaluation results to visualize. Run evaluate() first.")
            return
        
        try:
            accuracies = [
                self.eval_results['neural']['accuracy'] * 100,
                self.eval_results['symbolic']['accuracy'] * 100,
                self.eval_results['nexus']['accuracy'] * 100
            ]
            plt.figure(figsize=(10, 6))
            plt.bar(['Neural', 'Symbolic', 'NEXUS'], accuracies, color=['blue', 'green', 'red'])
            plt.title('Accuracy Comparison')
            plt.ylabel('Accuracy (%)')
            plt.ylim(0, 100)
            for i, v in enumerate(accuracies):
                plt.text(i, v + 1, f"{v:.1f}%", ha='center')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()
        except Exception as e:
            print(f"Error in accuracy plot: {e}")
        
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            models = ['neural', 'symbolic', 'nexus']
            titles = ['Neural Model', 'Symbolic Model', 'NEXUS Model']
            
            for i, (model, title) in enumerate(zip(models, titles)):
                confusion = self.eval_results[model]['confusion']
                row_sums = confusion.sum(axis=1, keepdims=True)
                norm_confusion = np.where(row_sums == 0, 0, confusion / row_sums)
                sns.heatmap(norm_confusion, annot=True, fmt='.2f', cmap='Blues', 
                            xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[i])
                axes[i].set_title(title)
                axes[i].set_ylabel('True Label' if i == 0 else '')
                axes[i].set_xlabel('Predicted Label')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error in confusion matrix plot: {e}")
        
        try:
            agreement = self.eval_results['agreement_cases']
            labels = ['All Correct', 'Neural Only', 'Symbolic Only', 'NEXUS Better', 'All Wrong']
            values = [agreement['all_correct'], agreement['neural_only'], agreement['symbolic_only'], 
                      agreement['nexus_better'], agreement['all_wrong']]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(labels, values, color=['green', 'blue', 'orange', 'red', 'gray'])
            plt.title('Model Agreement Analysis')
            plt.ylabel('Number of Cases')
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f"{int(height)}", ha='center', va='bottom')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()
        except Exception as e:
            print(f"Error in agreement plot: {e}")
        
        try:
            print("\nClass-wise Performance:")
            class_results = []
            for c in range(self.num_classes):
                class_name = self.class_names[c]
                metrics = {}
                for model in models:
                    true_labels = np.array(self.eval_results[model]['true_labels'])
                    predictions = np.array(self.eval_results[model]['predictions'])
                    tp = np.sum((predictions == c) & (true_labels == c))
                    fp = np.sum((predictions == c) & (true_labels != c))
                    fn = np.sum((predictions != c) & (true_labels == c))
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    metrics[model] = {'precision': precision, 'recall': recall, 'f1': f1}
                class_results.append([class_name, f"{metrics['neural']['f1']:.2f}", 
                                     f"{metrics['symbolic']['f1']:.2f}", f"{metrics['nexus']['f1']:.2f}"])
            
            print(tabulate(class_results, headers=['Class', 'Neural F1', 'Symbolic F1', 'NEXUS F1'], tablefmt='grid'))
        except Exception as e:
            print(f"Error in class-wise performance table: {e}")
        
        summary = {
            'neural_accuracy': self.eval_results['neural']['accuracy'],
            'symbolic_accuracy': self.eval_results['symbolic']['accuracy'],
            'nexus_accuracy': self.eval_results['nexus']['accuracy'],
            'agreement_cases': self.eval_results['agreement_cases']
        }
        return summary

# ===========================
# 6. Synthetic Medical Data
# ===========================
class SyntheticMedicalDataset:
    def __init__(self, num_samples=100, num_features=10, num_classes=4, random_state=42):  # Updated to 100 cases
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_classes = num_classes
        np.random.seed(random_state)
        self.X, self.y, self.symptom_dict = self._generate_data()
        
    def _generate_data(self):
        self.feature_names = [
            'fever', 'cough', 'runny_nose', 'sore_throat', 'fatigue',
            'shortness_of_breath', 'headache', 'muscle_pain', 'loss_of_taste', 'chills'
        ]
        X = np.zeros((self.num_samples, self.num_features), dtype=np.float32)
        y = np.zeros(self.num_samples, dtype=int)
        symptom_dict = {}
        
        # Balanced class distribution: 25 samples per class
        class_assignments = [0] * 25 + [1] * 25 + [2] * 25 + [3] * 25
        np.random.shuffle(class_assignments)
        y[:] = class_assignments
        
        for i in range(self.num_samples):
            active_symptoms = []
            class_id = y[i]
            
            if class_id == 0:  # Common Cold
                X[i, 2] = 1  # runny_nose
                active_symptoms.append('runny_nose')
                if np.random.random() > 0.3:
                    X[i, 3] = 1  # sore_throat
                    active_symptoms.append('sore_throat')
                if np.random.random() > 0.5:
                    X[i, 1] = 1  # cough
                    active_symptoms.append('cough')
                if np.random.random() > 0.7:
                    X[i, 0] = 1  # fever (less common)
                    active_symptoms.append('fever')
                    
            elif class_id == 1:  # Flu
                X[i, 0] = 1  # fever
                active_symptoms.append('fever')
                X[i, 1] = 1  # cough
                active_symptoms.append('cough')
                if np.random.random() > 0.4:
                    X[i, 4] = 1  # fatigue
                    active_symptoms.append('fatigue')
                if np.random.random() > 0.6:
                    X[i, 7] = 1  # muscle_pain
                    active_symptoms.append('muscle_pain')
                    
            elif class_id == 2:  # COVID
                X[i, 0] = 1  # fever
                active_symptoms.append('fever')
                X[i, 5] = 1  # shortness_of_breath
                active_symptoms.append('shortness_of_breath')
                if np.random.random() > 0.3:
                    X[i, 8] = 1  # loss_of_taste
                    active_symptoms.append('loss_of_taste')
                if np.random.random() > 0.5:
                    X[i, 1] = 1  # cough
                    active_symptoms.append('cough')
                    
            elif class_id == 3:  # Pneumonia
                X[i, 1] = 1  # cough
                active_symptoms.append('cough')
                X[i, 5] = 1  # shortness_of_breath
                active_symptoms.append('shortness_of_breath')
                X[i, 0] = 1  # fever
                active_symptoms.append('fever')
                if np.random.random() > 0.4:
                    X[i, 4] = 1  # fatigue
                    active_symptoms.append('fatigue')
                    
            symptom_dict[i] = active_symptoms
        
        return X, y, symptom_dict
    
    def get_dataloader(self, batch_size=1, train_indices=None, test_indices=None):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        
        if train_indices is not None and test_indices is not None:
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
            return train_loader, test_loader
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

# ===========================
# 7. Toy Experiment Setup
# ===========================
if __name__ == "__main__":
    # Define constants
    input_dim = 10
    num_classes = 4
    num_symbols = 10
    symbol_names = [
        'fever', 'cough', 'runny_nose', 'sore_throat', 'fatigue',
        'shortness_of_breath', 'headache', 'muscle_pain', 'loss_of_taste', 'chills'
    ]
    class_names = ['Common Cold', 'Flu', 'COVID', 'Pneumonia']

    # Initialize dataset with 100 cases
    dataset = SyntheticMedicalDataset(num_samples=100, num_features=input_dim, num_classes=num_classes)

    # Split into train (75%) and test (25%)
    indices = list(range(dataset.num_samples))
    np.random.shuffle(indices)
    split = int(0.75 * len(indices))  # 75 train, 25 test
    train_indices, test_indices = indices[:split], indices[split:]
    train_loader, test_loader = dataset.get_dataloader(batch_size=1, train_indices=train_indices, test_indices=test_indices)

    # Initialize NEXUS model
    nexus = NEXUSModel(input_dim, num_classes, num_symbols, symbol_names, class_names)

    # Initialize knowledge graph with improved rules
    kg = nexus.init_knowledge_graph()
    # Relations
    kg.add_relation(0, 'indicates', 1, 0.8)  # fever -> cough
    kg.add_relation(5, 'indicates', 3, 0.85)  # shortness_of_breath -> Pneumonia (class 3)
    # Rules directly to classes
    kg.add_rule([2], 0, 0.90)  # IF runny_nose THEN Common Cold
    kg.add_rule([2, 1], 0, 0.95)  # IF runny_nose AND cough THEN Common Cold
    kg.add_rule([0, 1], 1, 0.90)  # IF fever AND cough THEN Flu
    kg.add_rule([0, 1, 7], 1, 0.95)  # IF fever AND cough AND muscle_pain THEN Flu
    kg.add_rule([8, 5], 2, 0.95)  # IF loss_of_taste AND shortness_of_breath THEN COVID
    kg.add_rule([5, 0, 1], 3, 0.90)  # IF shortness_of_breath AND fever AND cough THEN Pneumonia
    kg.add_rule([5, 0, 4], 3, 0.92)  # IF shortness_of_breath AND fever AND fatigue THEN Pneumonia

    # Define symbol-to-class mapping with refined weights
    symbol_to_class_dict = {
        0: {1: 0.9, 2: 0.7, 3: 0.8},  # fever -> Flu, COVID, Pneumonia
        1: {0: 0.8, 1: 0.85, 3: 0.9}, # cough -> Common Cold, Flu, Pneumonia
        2: {0: 0.95},                 # runny_nose -> Common Cold
        3: {0: 0.7},                  # sore_throat -> Common Cold
        4: {1: 0.8, 3: 0.75},         # fatigue -> Flu, Pneumonia
        5: {2: 0.85, 3: 0.9},         # shortness_of_breath -> COVID, Pneumonia
        7: {1: 0.9},                  # muscle_pain -> Flu
        8: {2: 0.95}                  # loss_of_taste -> COVID
    }
    nexus.interface.set_symbol_to_class_mapping(symbol_to_class_dict)

    # Train neural component on training set
    print("Training Neural Component on Training Set:")
    nexus.train_neural(train_loader, num_epochs=5, lr=0.001)

    # Evaluate model on test set
    print("\nEvaluating Model on Test Set:")
    results = nexus.evaluate(test_loader, dataset.symptom_dict)

    # Visualize results
    print("\nVisualizing Results:")
    summary = nexus.visualize_results()

    # Detailed printout of test set results
    print("\n=== Detailed Results for Each Test Patient ===")
    print("This section compares Neural, Symbolic, and NEXUS predictions for the 25 test patients.")
    print("Each patient has a true diagnosis based on the synthetic dataset, and we evaluate generalization.")
    print("Key focus: 'shortness_of_breath' is a critical symptom for COVID and Pneumonia.\n")
    
    for i, (inputs, labels) in enumerate(test_loader):
        test_idx = test_indices[i]
        try:
            result = nexus.diagnose(inputs, dataset.symptom_dict[test_idx])
            true_label = class_names[labels.item()]
            print(f"Test Patient {i+1} (Index {test_idx}):")
            print(f"True Diagnosis: {true_label}")
            print(f"Symptoms: {', '.join(dataset.symptom_dict[test_idx])}")
            print(f"Neural Prediction: {result['neural']['class_name']} (Confidence: {result['neural']['confidence']:.2f})")
            print(f"Symbolic Prediction: {result['symbolic']['class_name']} (Confidence: {result['symbolic']['confidence']:.2f})")
            print(f"NEXUS Prediction: {result['nexus']['class_name']} (Confidence: {result['nexus']['confidence']:.2f})")
            print(f"Detailed Explanation:\n{nexus.explain_diagnosis(result, detail_level='high')}")
            print("-" * 80)
        except Exception as e:
            print(f"Error processing Test Patient {i+1}: {e}")
            print("-" * 80)
    
    # Summary explanation
    print("\n=== Summary Explanation ===")
    print(f"Overall Performance on Test Set:")
    print(f"- Neural Accuracy: {summary['neural_accuracy']*100:.1f}%")
    print(f"- Symbolic Accuracy: {summary['symbolic_accuracy']*100:.1f}%")
    print(f"- NEXUS Accuracy: {summary['nexus_accuracy']*100:.1f}%")
    print("\nAgreement Analysis:")
    print(f"- All Correct: {summary['agreement_cases']['all_correct']} cases")
    print(f"- Neural Only Correct: {summary['agreement_cases']['neural_only']} cases")
    print(f"- Symbolic Only Correct: {summary['agreement_cases']['symbolic_only']} cases")
    print(f"- NEXUS Better: {summary['agreement_cases']['nexus_better']} cases")
    print(f"- All Wrong: {summary['agreement_cases']['all_wrong']} cases")
    print("\nInsights:")
    print("- Neural: Relies on pattern recognition from 75 training cases. Tested on 25 unseen test cases.")
    print("- Symbolic: Enhanced with comprehensive rules and direct class scoring for better accuracy.")
    print("- NEXUS: Balances improved symbolic reasoning with neural predictions across a larger dataset.")
    print("This updated model scales to 100 cases to improve training and generalization assessment.")