import networkx as nx
import pandas as pd
import ast
from sklearn.base import BaseEstimator, TransformerMixin

class CentralityFeatures(BaseEstimator, TransformerMixin):
    """Class to compute centrality features for nodes in a graph"""
    
    def __init__(self, is_test=False):
        self.is_test = is_test
        self.feature_names = [
            'degree', 'closeness', 'betweenness', 
            'pagerank', 'eigencentrality','eccentricity',
            'is_articulation'
        ]
    
    def _compute_centralities(self, edgelist):
        """Compute all centrality metrics for a graph"""
        T = nx.from_edgelist(edgelist)
        
        dc = nx.degree_centrality(T)
        cc = nx.closeness_centrality(T)
        bc = nx.betweenness_centrality(T)
        pr = nx.pagerank(T)
        ec = nx.eigenvector_centrality(T, max_iter=1000, tol=1e-04)
        et = nx.eccentricity(T)
        ap = list(nx.articulation_points(T))
        
        return {v: (dc[v], cc[v], bc[v], pr[v], ec[v], 
                    et[v], 1 if v in ap else 0) 
                for v in T}
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """X should be a DataFrame with columns: ['language', 'sentence', 'edgelist', 'root']"""
        all_rows = []
        
        for _, row in X.iterrows():
            language = row['language']
            sentence_id = row['sentence']
            n = row['n']  
            edgelist = ast.literal_eval(row['edgelist'])
            root = int(row['root']) if not self.is_test else None
            cent_dict = self._compute_centralities(edgelist)
            
            for vertex, features in cent_dict.items():
                row_data = {
                    'language': language,
                    'sentence': sentence_id,
                    'n': n,
                    'vertex': vertex,
                }
                if self.is_test:
                    row_data['id'] = row['id']
                else:
                    row_data['is_root'] = 1 if vertex == root else 0
                
                row_data['edgelist'] = row['edgelist']
                # Add all centrality features
                row_data.update(zip(self.feature_names, features))
                all_rows.append(row_data)
        
        return pd.DataFrame(all_rows)

class StructuralFeatures(BaseEstimator, TransformerMixin):
    """Class to compute structural features for nodes in a graph"""
    
    def __init__(self):
        self.graph_dict = {}
        self.feature_names = ['max_depth', 'avg_depth', 'subtree_size']
    
    def _max_depth_from_node(self, G, node):
        """Longest directed path from node to any leaf in its subtree."""
        if G.out_degree(node) == 0:  # Leaf node
            return 0
        return 1 + max(self._max_depth_from_node(G, child) 
                      for child in G.successors(node))
    
    def _avg_depth_from_node(self, G, node):
        """Mean directed distance to leaves in the subtree."""
        leaves = [n for n in nx.descendants(G, node) if G.out_degree(n) == 0]
        if not leaves:
            return 0  # Node is a leaf
        distances = [nx.shortest_path_length(G, node, leaf) for leaf in leaves]
        return sum(distances) / len(distances)
    
    def _subtree_size(self, G, node):
        """Number of nodes in the subtree rooted at this node (including self)"""
        return 1 + len(nx.descendants(G, node))
    
    def _compute_features(self, edgelist, vertex):
        
        edges = ast.literal_eval(edgelist)
        G = nx.DiGraph(edges)
        
        if not G.has_node(vertex):
            return [0, 0, 0]
            
        return [
            self._max_depth_from_node(G, vertex),
            self._avg_depth_from_node(G, vertex),
            self._subtree_size(G, vertex)
        ]
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        results = []
        
        for _, row in X.iterrows():
            features = self._compute_features(
                row['edgelist'], 
                row['vertex']
            )
            results.append(pd.Series(features, index=self.feature_names))
        
        return pd.concat(results, axis=1).transpose()
    
class GraphFeaturePipeline(BaseEstimator, TransformerMixin):
    def __init__(self, is_test=False):
        self.is_test = is_test
        self.centrality = CentralityFeatures(is_test=self.is_test)
        self.structural = StructuralFeatures()
    
    def fit(self, X, y=None):
        if not self.is_test:
            self.structural.fit(X)
            self.centrality.fit(X)
        return self
    
    def transform(self, X):
        centrality_df = self.centrality.transform(X)
        structural_df = self.structural.transform(centrality_df)
        combined_df = pd.concat([centrality_df, structural_df], axis=1)
        # Drop 'edgelist' column from final output
        if 'edgelist' in combined_df.columns:
            combined_df = combined_df.drop(columns=['edgelist'])
        return combined_df
    
if __name__ == "__main__":
    train_data = pd.read_csv('../data/train.csv')
    pipeline = GraphFeaturePipeline()
    train_processed = pipeline.fit_transform(train_data)
    train_processed.to_csv('../data/train_processed.csv', index=False)
    print(f'Train Data: {train_processed.head(10)}')
    test_data = pd.read_csv('../data/test.csv')
    pipeline = GraphFeaturePipeline(is_test=True)
    test_processed = pipeline.transform(test_data)
    test_processed.to_csv('../data/test_processed.csv', index=False)
    print(f'Test Data: {test_processed.head(10)}')