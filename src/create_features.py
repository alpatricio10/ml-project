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
        self.feature_names = ['subtree_size']
    
    def _calculate_subtree_sizes(self, edgelist):
        edges = ast.literal_eval(edgelist)
        children_map = {}
        for parent, child in edges:
            children_map.setdefault(parent, []).append(child)
        
        cache = {}
        def _subtree_size(node):
            if node in cache:
                return cache[node]
            size = 1 + sum(_subtree_size(child) for child in children_map.get(node, []))
            cache[node] = size
            return size
        
        nodes = set(n for edge in edges for n in edge)
        return {node: _subtree_size(node) for node in nodes}
    
    def fit(self, X, y=None):
        return self
  
    def transform(self, X):
        results = []
        for _, row in X.iterrows():
            edgelist = row['edgelist']
            vertex = row['vertex']
            
            subtree_sizes = self._calculate_subtree_sizes(edgelist)
            size = subtree_sizes.get(vertex, 0)
            results.append(pd.Series([size], index=self.feature_names))
        
        # Return DataFrame with same index as input
        return pd.DataFrame(results, index=X.index)

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