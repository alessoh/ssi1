import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import networkx as nx

# ======================
# 1. Enhanced Dataset with Comorbidities
# ======================
class MedicalDataset:
    def __init__(self, n_samples=1000, imbalance_ratio=(0.70, 0.20, 0.10)):
        # Features: [fever, cough, fatigue, shortness_of_breath, immunocompromised, elderly]
        self.feature_names = ["fever", "cough", "fatigue", "shortness_of_breath", "immunocompromised", "elderly"]
        
        n_mild = int(n_samples * imbalance_ratio[0])
        n_pneumonia = int(n_samples * imbalance_ratio[1])
        n_severe = n_samples - n_mild - n_pneumonia
        
        # Generate basic symptom data
        mild_data = np.random.rand(n_mild, 4) * [3, 5, 5, 5] + [97, 0, 0, 0]
        pneumonia_data = np.random.rand(n_pneumonia, 4) * [3, 5, 5, 5] + [98, 5, 5, 5]
        pneumonia_data[:, 3] = np.random.rand(n_pneumonia) * 5 + 7  # Higher shortness of breath
        severe_data = np.random.rand(n_severe, 4) * [3, 5, 5, 5] + [101, 7, 5, 5]
        
        # Add comorbidity factors (immunocompromised, elderly)
        mild_comorbidities = np.random.rand(n_mild, 2) < 0.1  # 10% chance of comorbidity
        pneumonia_comorbidities = np.random.rand(n_pneumonia, 2) < 0.3  # 30% chance of comorbidity
        severe_comorbidities = np.random.rand(n_severe, 2) < 0.5  # 50% chance of comorbidity
        
        # Combine symptoms with comorbidities
        self.data = np.vstack([
            np.hstack([mild_data, mild_comorbidities.astype(float)]),
            np.hstack([pneumonia_data, pneumonia_comorbidities.astype(float)]),
            np.hstack([severe_data, severe_comorbidities.astype(float)])
        ])
        
        self.labels = np.concatenate([
            np.zeros(n_mild),
            np.ones(n_pneumonia),
            np.full(n_severe, 2)
        ]).astype(int)
        
        self.label_names = ['mild', 'pneumonia', 'severe']

# ======================
# 2. Enhanced Neural Component with Feature Extraction
# ======================
class EnhancedNeuralDiagnoser(nn.Module):
    def __init__(self, input_dim=6, hidden_dims=[32, 16], output_dim=3):
        super().__init__()
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = dim
            
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, output_dim)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x):
        return self.feature_extractor(x)

# ======================
# 3. Knowledge Graph Symbolic Component
# ======================
class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        
        # Add symptom nodes
        self.graph.add_node("fever", type="symptom")
        self.graph.add_node("high_fever", type="symptom_level", threshold=101)
        self.graph.add_node("moderate_fever", type="symptom_level", threshold=99)
        
        self.graph.add_node("cough", type="symptom")
        self.graph.add_node("severe_cough", type="symptom_level", threshold=7)
        self.graph.add_node("moderate_cough", type="symptom_level", threshold=4)
        
        self.graph.add_node("fatigue", type="symptom")
        self.graph.add_node("severe_fatigue", type="symptom_level", threshold=7)
        
        self.graph.add_node("shortness_of_breath", type="symptom")
        self.graph.add_node("severe_sob", type="symptom_level", threshold=7)
        
        # Add comorbidity nodes
        self.graph.add_node("immunocompromised", type="comorbidity")
        self.graph.add_node("elderly", type="comorbidity")
        
        # Add disease nodes
        self.graph.add_node("mild", type="disease")
        self.graph.add_node("pneumonia", type="disease")
        self.graph.add_node("severe", type="disease")
        
        # Connect symptoms to levels
        self.graph.add_edge("fever", "high_fever")
        self.graph.add_edge("fever", "moderate_fever")
        self.graph.add_edge("cough", "severe_cough")
        self.graph.add_edge("cough", "moderate_cough")
        self.graph.add_edge("fatigue", "severe_fatigue")
        self.graph.add_edge("shortness_of_breath", "severe_sob")
        
        # Connect symptoms to diseases
        self.graph.add_edge("high_fever", "severe", weight=0.7)
        self.graph.add_edge("moderate_fever", "pneumonia", weight=0.4)
        self.graph.add_edge("severe_cough", "severe", weight=0.6)
        self.graph.add_edge("moderate_cough", "pneumonia", weight=0.5)
        self.graph.add_edge("severe_fatigue", "pneumonia", weight=0.4)
        self.graph.add_edge("severe_sob", "pneumonia", weight=0.8)
        
        # Connect comorbidities to increased risk
        self.graph.add_edge("immunocompromised", "pneumonia", weight=0.3, risk_multiplier=2.0)
        self.graph.add_edge("immunocompromised", "severe", weight=0.5, risk_multiplier=3.0)
        self.graph.add_edge("elderly", "pneumonia", weight=0.3, risk_multiplier=1.5)
        self.graph.add_edge("elderly", "severe", weight=0.4, risk_multiplier=2.0)
        
    def reason(self, case, feature_names=None):
        """Perform graph-based reasoning on patient symptoms"""
        if feature_names is None:
            feature_names = ["fever", "cough", "fatigue", "shortness_of_breath", 
                            "immunocompromised", "elderly"]
        
        # Initialize disease scores
        disease_scores = {"mild": 0.3, "pneumonia": 0, "severe": 0}
        
        # Evaluate symptoms
        if case[0] > self.graph.nodes["high_fever"]["threshold"]:
            disease_scores["severe"] += self.graph.edges["high_fever", "severe"]["weight"]
        elif case[0] > self.graph.nodes["moderate_fever"]["threshold"]:
            disease_scores["pneumonia"] += self.graph.edges["moderate_fever", "pneumonia"]["weight"]
            
        if case[1] > self.graph.nodes["severe_cough"]["threshold"]:
            disease_scores["severe"] += self.graph.edges["severe_cough", "severe"]["weight"]
        elif case[1] > self.graph.nodes["moderate_cough"]["threshold"]:
            disease_scores["pneumonia"] += self.graph.edges["moderate_cough", "pneumonia"]["weight"]
            
        if case[2] > self.graph.nodes["severe_fatigue"]["threshold"]:
            disease_scores["pneumonia"] += self.graph.edges["severe_fatigue", "pneumonia"]["weight"]
            
        if case[3] > self.graph.nodes["severe_sob"]["threshold"]:
            disease_scores["pneumonia"] += self.graph.edges["severe_sob", "pneumonia"]["weight"]
            
        # Apply comorbidity factors
        reasoning_steps = []
        
        if case[4] > 0.5:  # Immunocompromised
            for disease in ["pneumonia", "severe"]:
                risk_multiplier = self.graph.edges["immunocompromised", disease]["risk_multiplier"]
                old_score = disease_scores[disease]
                disease_scores[disease] *= risk_multiplier
                reasoning_steps.append(f"Immunocompromised: {disease} risk {old_score:.2f} → {disease_scores[disease]:.2f}")
                
        if case[5] > 0.5:  # Elderly
            for disease in ["pneumonia", "severe"]:
                risk_multiplier = self.graph.edges["elderly", disease]["risk_multiplier"]
                old_score = disease_scores[disease]
                disease_scores[disease] *= risk_multiplier
                reasoning_steps.append(f"Elderly: {disease} risk {old_score:.2f} → {disease_scores[disease]:.2f}")
        
        # Normalize scores
        max_score = max(disease_scores.values())
        for disease in disease_scores:
            disease_scores[disease] /= max_score
        
        # Get prediction and confidence
        prediction = max(disease_scores.items(), key=lambda x: x[1])
        disease_name, confidence = prediction
        
        # Map disease name to label index
        label_map = {"mild": 0, "pneumonia": 1, "severe": 2}
        
        return label_map[disease_name], confidence, reasoning_steps, disease_scores

    def update_from_feedback(self, case, true_label, prediction, confidence, label_names):
        """Update the knowledge graph based on feedback from errors"""
        if confidence < 0.7:  # Only update on low confidence predictions
            label_map = {0: "mild", 1: "pneumonia", 2: "severe"}
            true_disease = label_map[true_label]
            
            # Identify the most significant symptom
            symptoms = ["fever", "cough", "fatigue", "shortness_of_breath"]
            symptom_levels = ["high_fever", "severe_cough", "severe_fatigue", "severe_sob"]
            max_symptom_idx = np.argmax(case[:4])
            symptom = symptoms[max_symptom_idx]
            symptom_level = symptom_levels[max_symptom_idx]
            
            # Strengthen the connection to the true disease
            if self.graph.has_edge(symptom_level, true_disease):
                current_weight = self.graph.edges[symptom_level, true_disease]["weight"]
                self.graph.edges[symptom_level, true_disease]["weight"] = min(1.0, current_weight + 0.05)
            else:
                self.graph.add_edge(symptom_level, true_disease, weight=0.4)
                
            # Update comorbidity risk if present
            if case[4] > 0.5 and true_label > 0:  # Immunocompromised with pneumonia or severe
                current_mult = self.graph.edges["immunocompromised", true_disease]["risk_multiplier"]
                self.graph.edges["immunocompromised", true_disease]["risk_multiplier"] = current_mult + 0.1
                
            if case[5] > 0.5 and true_label > 0:  # Elderly with pneumonia or severe
                current_mult = self.graph.edges["elderly", true_disease]["risk_multiplier"]
                self.graph.edges["elderly", true_disease]["risk_multiplier"] = current_mult + 0.1
            
            return True  # Knowledge graph was updated
        
        return False  # No update performed

# ======================
# 4. Integrated Neural-Symbolic NEXUS Model
# ======================
class NEXUSModel:
    def __init__(self, neural_model, knowledge_graph, feature_names):
        self.neural = neural_model
        self.kg = knowledge_graph
        self.softmax = nn.Softmax(dim=1)
        self.feature_names = feature_names
        self.learning_history = []
        
    def predict(self, x):
        # Neural processing
        x_tensor = torch.FloatTensor(x)
        neural_features = self.neural.extract_features(x_tensor)
        neural_out = self.neural.classifier(neural_features)
        neural_probs = self.softmax(neural_out).detach().numpy()
        neural_conf = np.max(neural_probs, axis=1)
        neural_preds = np.argmax(neural_probs, axis=1)
        
        # Symbolic knowledge-graph reasoning
        symbolic_preds = []
        symbolic_confs = []
        reasoning_paths = []
        disease_scores_list = []
        
        for case in x:
            pred, conf, reasoning, scores = self.kg.reason(case, self.feature_names)
            symbolic_preds.append(pred)
            symbolic_confs.append(conf)
            reasoning_paths.append(reasoning)
            disease_scores_list.append(scores)
        
        # Metacognitive integration
        combined = []
        explanations = []
        for idx, (n_conf, n_pred, s_pred, s_conf, reasoning, scores) in enumerate(
            zip(neural_conf, neural_preds, symbolic_preds, symbolic_confs, reasoning_paths, disease_scores_list)):
            
            # Determine confidence factors
            neural_certainty = n_conf
            symbolic_certainty = s_conf
            
            # Check if neural and symbolic agree
            if n_pred == s_pred:
                final_pred = n_pred
                combined_conf = max(n_conf, s_conf)
                exp = f"Neural and Symbolic agree: {final_pred} (conf: {combined_conf:.2f})"
            else:
                # Look for risk factors
                risk_factors = []
                if x[idx, 4] > 0.5:  # immunocompromised
                    risk_factors.append("immunocompromised")
                if x[idx, 5] > 0.5:  # elderly
                    risk_factors.append("elderly")
                
                # High-risk cases should favor symbolic
                if risk_factors and s_pred > n_pred:  # Symbolic predicts more severe condition
                    final_pred = s_pred
                    exp = f"Symbolic override due to risk factors ({', '.join(risk_factors)})"
                # Very confident neural prediction
                elif n_conf > 0.85:
                    final_pred = n_pred
                    exp = f"Neural dominant ({n_conf:.2f} > 0.85)"
                # Confident symbolic prediction
                elif s_conf > 0.75:
                    final_pred = s_pred
                    exp = f"Symbolic dominant ({s_conf:.2f} > 0.75)"
                # Neural has higher confidence
                elif n_conf > s_conf + 0.1:
                    final_pred = n_pred
                    exp = f"Neural preferred ({n_conf:.2f} > {s_conf:.2f}+0.1)"
                # Symbolic has higher confidence
                elif s_conf > n_conf:
                    final_pred = s_pred
                    exp = f"Symbolic preferred ({s_conf:.2f} > {n_conf:.2f})"
                # Default to neural
                else:
                    final_pred = n_pred
                    exp = f"Neural default ({n_conf:.2f} vs {s_conf:.2f})"
            
            # Append the reasoning steps to explanation
            if reasoning:
                exp += f" - Reasoning: {' | '.join(reasoning)}"
                
            combined.append(final_pred)
            explanations.append(exp)
            
        return combined, explanations, neural_preds, symbolic_preds, neural_conf, symbolic_confs
    
    def self_improve(self, x, true_labels):
        """Learn from successes and failures to improve both components"""
        combined_preds, _, neural_preds, symbolic_preds, neural_conf, symbolic_confs = self.predict(x)
        
        update_count = 0
        
        # Analyze each case
        for i, (true, pred, n_pred, s_pred, n_conf, s_conf) in enumerate(
            zip(true_labels, combined_preds, neural_preds, symbolic_preds, neural_conf, symbolic_confs)):
            
            # Check for failures where the final prediction was wrong
            if pred != true:
                # If neural was correct but symbolic wasn't trusted
                if n_pred == true and s_pred != true:
                    update = self.kg.update_from_feedback(x[i], true, s_pred, s_conf, ["mild", "pneumonia", "severe"])
                    if update:
                        update_count += 1
                        self.learning_history.append({
                            'type': 'symbolic_update',
                            'case': x[i].tolist(),
                            'true_label': true,
                            'neural_pred': n_pred,
                            'symbolic_pred': s_pred
                        })
        
        return update_count
    
    def get_learning_history(self):
        return self.learning_history

# ======================
# 5. Advanced Visualization Tools
# ======================
def plot_decision_analysis(nexus_model, test_data, test_labels, feature_names):
    """Visualize the decision-making process of the NEXUS model"""
    preds, explanations, neural_preds, symbolic_preds, neural_conf, symbolic_confs = nexus_model.predict(test_data)
    
    # Calculate accuracy metrics
    neural_accuracy = accuracy_score(test_labels, neural_preds)
    symbolic_accuracy = accuracy_score(test_labels, symbolic_preds)
    nexus_accuracy = accuracy_score(test_labels, preds)
    
    # Create confusion matrix data
    agreement = sum(n == s for n, s in zip(neural_preds, symbolic_preds)) / len(neural_preds)
    
    # Identify different cases
    both_correct = [(i, test_data[i]) for i in range(len(test_data)) 
                   if neural_preds[i] == test_labels[i] and symbolic_preds[i] == test_labels[i]]
    
    neural_correct = [(i, test_data[i]) for i in range(len(test_data))
                     if neural_preds[i] == test_labels[i] and symbolic_preds[i] != test_labels[i]]
    
    symbolic_correct = [(i, test_data[i]) for i in range(len(test_data))
                       if neural_preds[i] != test_labels[i] and symbolic_preds[i] == test_labels[i]]
    
    both_wrong = [(i, test_data[i]) for i in range(len(test_data))
                 if neural_preds[i] != test_labels[i] and symbolic_preds[i] != test_labels[i]]
    
    print(f"=== Decision Analysis ===")
    print(f"Neural Accuracy: {neural_accuracy:.4f}")
    print(f"Symbolic Accuracy: {symbolic_accuracy:.4f}")
    print(f"NEXUS Accuracy: {nexus_accuracy:.4f}")
    print(f"Neural-Symbolic Agreement Rate: {agreement:.4f}")
    print(f"Both correct: {len(both_correct)} cases")
    print(f"Neural correct only: {len(neural_correct)} cases")
    print(f"Symbolic correct only: {len(symbolic_correct)} cases")
    print(f"Both wrong: {len(both_wrong)} cases")
    
    # Plot feature distributions for each scenario
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    scenarios = [
        (both_correct[:10], "Both Correct", axes[0, 0]),
        (neural_correct[:10], "Neural Correct Only", axes[0, 1]),
        (symbolic_correct[:10], "Symbolic Correct Only", axes[1, 0]),
        (both_wrong[:10], "Both Wrong", axes[1, 1])
    ]
    
    for items, title, ax in scenarios:
        if not items:
            ax.text(0.5, 0.5, f"No {title} cases", ha='center')
            ax.set_title(title)
            continue
            
        data = np.array([test_data[i] for i, _ in items])
        ax.boxplot([data[:, i] for i in range(data.shape[1])], labels=feature_names)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Show some detailed explanations
    print("\n=== Sample Explanations ===")
    for s in range(4):
        scenario_name = ["Both Correct", "Neural Correct Only", "Symbolic Correct Only", "Both Wrong"][s]
        items = [both_correct, neural_correct, symbolic_correct, both_wrong][s]
        
        if items:
            print(f"\n{scenario_name} Examples:")
            for i, (idx, _) in enumerate(items[:3]):
                print(f"  Case {i+1}: {explanations[idx]}")
                print(f"    Features: {test_data[idx]}")
                print(f"    True: {test_labels[idx]} Neural: {neural_preds[idx]} Symbolic: {symbolic_preds[idx]}")

def visualize_knowledge_graph(kg):
    """Visualize the medical knowledge graph"""
    plt.figure(figsize=(12, 10))
    
    pos = nx.spring_layout(kg.graph, seed=42)
    
    # Define node colors by type
    node_colors = []
    for node in kg.graph.nodes():
        if kg.graph.nodes[node]['type'] == 'disease':
            node_colors.append('red')
        elif kg.graph.nodes[node]['type'] == 'symptom':
            node_colors.append('blue')
        elif kg.graph.nodes[node]['type'] == 'symptom_level':
            node_colors.append('green')
        elif kg.graph.nodes[node]['type'] == 'comorbidity':
            node_colors.append('purple')
        else:
            node_colors.append('gray')
    
    # Define edge weights for visualization
    edge_weights = [kg.graph[u][v].get('weight', 0.5) * 2 for u, v in kg.graph.edges()]
    
    # Draw the graph
    nx.draw(kg.graph, pos, with_labels=True, node_color=node_colors, 
            node_size=1500, font_size=10, font_weight='bold', 
            edge_color='gray', width=edge_weights, alpha=0.7)
    
    plt.title("Medical Knowledge Graph", fontsize=15)
    plt.tight_layout()
    plt.show()

# ======================
# 6. Training and Evaluation Workflow
# ======================
def run_experiment():
    """Run a full experiment with the NEXUS architecture"""
    # Create dataset
    dataset = MedicalDataset(4000, imbalance_ratio=(0.65, 0.25, 0.10))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.labels, test_size=0.2, random_state=42)
    
    # Initialize components
    neural_net = EnhancedNeuralDiagnoser(input_dim=6, hidden_dims=[32, 16], output_dim=3)
    knowledge_graph = MedicalKnowledgeGraph()
    nexus = NEXUSModel(neural_net, knowledge_graph, dataset.feature_names)
    
    # Train neural component
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Convert training data to tensors
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    
    # Training loop with early stopping
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(200):
        neural_net.train()
        optimizer.zero_grad()
        
        outputs = neural_net(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        # Check for early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Evaluate components
    neural_net.eval()
    
    # Baseline neural network evaluation
    with torch.no_grad():
        neural_outputs = neural_net(torch.FloatTensor(X_test))
        neural_preds = torch.argmax(neural_outputs, dim=1).numpy()
    
    print("\n=== Neural Network Only ===")
    print(classification_report(y_test, neural_preds, target_names=dataset.label_names))
    
    # NEXUS evaluation
    nexus_preds, explanations, _, _, _, _ = nexus.predict(X_test)
    
    print("\n=== NEXUS Model ===")
    print(classification_report(y_test, nexus_preds, target_names=dataset.label_names))
    
    # Self-improvement phase
    print("\n=== Self-Improvement Phase ===")
    for i in range(3):
        updates = nexus.self_improve(X_test, y_test)
        print(f"Iteration {i+1}: Made {updates} knowledge graph updates")
        
        # Re-evaluate after improvement
        nexus_preds, explanations, _, _, _, _ = nexus.predict(X_test)
        print(f"Updated accuracy: {accuracy_score(y_test, nexus_preds):.4f}")
    
    # Visualize results
    plot_decision_analysis(nexus, X_test, y_test, dataset.feature_names)
    visualize_knowledge_graph(knowledge_graph)
    
    # Show some examples of interesting cases
    print("\n=== Interesting Cases ===")
    nexus_preds, explanations, neural_preds, symbolic_preds, neural_conf, symbolic_confs = nexus.predict(X_test)
    
    # Find cases where symbolic reasoning overrode neural and was correct (high-risk patients)
    override_cases = []
    for i in range(len(X_test)):
        if (neural_preds[i] != symbolic_preds[i] and 
            symbolic_preds[i] == y_test[i] and
            nexus_preds[i] == symbolic_preds[i] and
            (X_test[i, 4] > 0.5 or X_test[i, 5] > 0.5)):  # Has risk factors
            override_cases.append(i)
    
    for i, idx in enumerate(override_cases[:5]):
        print(f"\nCase {i+1}:")
        print(f"Symptoms: {X_test[idx]}")
        print(f"Risk factors: {'Immunocompromised' if X_test[idx, 4] > 0.5 else ''} {'Elderly' if X_test[idx, 5] > 0.5 else ''}")
        print(f"True: {y_test[idx]} ({dataset.label_names[y_test[idx]]})")
        print(f"Neural: {neural_preds[idx]} ({dataset.label_names[neural_preds[idx]]}) - Conf: {neural_conf[idx]:.2f}")
        print(f"Symbolic: {symbolic_preds[idx]} ({dataset.label_names[symbolic_preds[idx]]}) - Conf: {symbolic_confs[idx]:.2f}")
        print(f"NEXUS: {nexus_preds[idx]} ({dataset.label_names[nexus_preds[idx]]})")
        print(f"Explanation: {explanations[idx]}")
    
    return nexus, dataset

# ======================
# 7. Main Execution
# ======================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    nexus_model, dataset = run_experiment()