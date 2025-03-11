import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class KnowledgeGraphEnhancer:
    def __init__(self, domain='medical'):
        self.graph = nx.DiGraph()
        self.domain = domain
        self.initialize_core_medical_knowledge()

    def initialize_core_medical_knowledge(self):
        # Base medical knowledge
        self.add_node('diabetes', 'disease', {'description': 'Diabetes mellitus'}, 0.9)
        self.add_node('hypertension', 'disease', {'description': 'High blood pressure'}, 0.9)
        self.add_node('asthma', 'disease', {'description': 'Chronic respiratory disease'}, 0.9)
        self.add_node('glucose_level', 'biomarker', {}, 0.8)
        self.add_node('systolic_bp', 'biomarker', {}, 0.8)
        self.add_node('wheezing_severity', 'symptom', {}, 0.8)

    def add_node(self, name, node_type, attributes=None, confidence=1.0):
        self.graph.add_node(name, 
                          type=node_type,
                          attributes=attributes or {},
                          confidence=confidence)

    def add_edge(self, source, target, relation, weight=1.0, confidence=1.0):
        self.graph.add_edge(source, target,
                          relation=relation,
                          weight=weight,
                          confidence=confidence)

    def extract_knowledge_from_text(self, text):
        nodes = []
        edges = []
        text = text.lower()

        # Detect biomarkers and symptoms
        biomarkers = {
            'glucose': ('glucose_level', 0.8),
            'blood pressure': ('systolic_bp', 0.7),
            'wheezing': ('wheezing_severity', 0.9)
        }

        # Detect diseases
        diseases = {
            'diabetes': ['glucose', 'insulin', 'thirst'],
            'hypertension': ['blood pressure', 'headache', 'stress'],
            'asthma': ['wheezing', 'breathing', 'inhaler']
        }

        # Create nodes and relationships
        for term, (node_name, conf) in biomarkers.items():
            if term in text:
                nodes.append((node_name, 'biomarker'))
                for disease, keywords in diseases.items():
                    if any(kw in text for kw in keywords):
                        edges.append((node_name, disease, 'indicates', conf))

        return nodes, edges

    def enhance_features(self, feature_names):
        # Create clinically meaningful interactions
        clinical_relationships = [
            ('glucose_level', 'diabetes', 0.8),
            ('systolic_bp', 'hypertension', 0.9),
            ('wheezing_severity', 'asthma', 0.85)
        ]

        for source, target, weight in clinical_relationships:
            if source in feature_names and target in self.graph.nodes:
                self.add_edge(source, target, 'clinical_relationship', weight)

        return self.graph

    def visualize(self, max_nodes=20):
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, 
               node_size=1200, font_size=8, 
               edge_color='gray', width=0.5)
        plt.title("Medical Knowledge Graph")
        plt.show()