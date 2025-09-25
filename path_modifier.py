# path_modifier.py
import networkx as nx
from typing import List, Dict

class PathModifier:
    def __init__(self, graph: nx.DiGraph):
        """
        Initialize the path modifier with a transportation graph.
        
        Args:
            graph: The transportation network graph (directed)
        """
        self.graph = graph
        
    def _calculate_path_cost(self, path: List[str]) -> float:
        """Calculate the total cost of a path"""
        if not path or len(path) < 2:
            return float('inf')
        
        cost = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if self.graph.has_edge(u, v):
                cost += self.graph[u][v].get('cost', 0)
        return cost

    def _calculate_path_time(self, path: List[str]) -> float:
        """Calculate the total time of a path including transfers"""
        if not path or len(path) < 2:
            return float('inf')
        
        total_time = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if self.graph.has_edge(u, v):
                total_time += self.graph[u][v]['time']
                
                # Add transfer time if mode changes
                if i < len(path) - 2:
                    next_edge = self.graph[v][path[i+2]]
                    if next_edge['mode'] != self.graph[u][v]['mode']:
                        transfer_time = (self.graph[u][v]['wait_time'] + next_edge['wait_time']) / 2
                        total_time += transfer_time
        return total_time

    def _calculate_path_distance(self, path: List[str]) -> float:
        """Calculate the total distance of a path"""
        if not path or len(path) < 2:
            return float('inf')
        
        distance = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if self.graph.has_edge(u, v):
                distance += self.graph[u][v]['distance']
        return distance

    def _get_path_modes(self, path: List[str]) -> List[str]:
        """Get the transportation modes used in a path"""
        if not path or len(path) < 2:
            return []
        
        modes = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if self.graph.has_edge(u, v):
                mode = self.graph[u][v]['mode']
                if not modes or modes[-1] != mode:
                    modes.append(mode)
        return modes

    def _modify_single_path(self, path: List[str]) -> List[Dict]:
        """
        Create modified versions of a single path by replacing edges with alternatives.
        
        Args:
            path: A list of node IDs representing the original path
            
        Returns:
            List of solution dictionaries including original and modified paths
        """
        solutions = [{
            'path': path,
            'cost': self._calculate_path_cost(path),
            'travel_time': self._calculate_path_time(path),
            'distance': self._calculate_path_distance(path),
            'modes': self._get_path_modes(path),
        }]
        
        # Only attempt modifications for paths with at least 3 nodes
        if len(path) < 3:
            return solutions
        
        # Create a subgraph containing only the nodes in the path
        path_nodes = set(path)
        subgraph = self.graph.subgraph(path_nodes).copy()
        
        # Remove original path edges
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if subgraph.has_edge(u, v):
                subgraph.remove_edge(u, v)
        
        # Try to find alternative paths in the modified subgraph
        try:
            # Find all simple paths between start and end nodes
            all_paths = list(nx.all_simple_paths(
                subgraph, 
                source=path[0], 
                target=path[-1],
                cutoff=len(path) + 2  # Allow slightly longer paths
            ))
            
            # Add valid alternative paths
            for alt_path in all_paths:
                if alt_path != path:  # Ensure it's different
                    solutions.append({
                        'path': alt_path,
                        'cost': self._calculate_path_cost(alt_path),
                        'travel_time': self._calculate_path_time(alt_path),
                        'distance': self._calculate_path_distance(alt_path),
                        'modes': self._get_path_modes(alt_path),
                        'algorithm': 'Modified Path',
                    })
                    
        except nx.NetworkXNoPath:
            # No alternative paths found
            pass
            
        return solutions

    def modify_solutions(self, solutions: List[Dict]) -> List[Dict]:
        """
        Process a list of solutions, modifying each path to create alternatives.
        
        Args:
            solutions: List of solution dictionaries, each containing a 'path' key
            
        Returns:
            List of solution dictionaries including originals and modifications
        """
        modified_solutions = []
        
        for solution in solutions:
            path = solution.get('path', [])
            if not path:
                continue
                
            # Preserve original solution metadata
            original_solution = solution.copy()
            
            # Create modified paths
            path_variations = self._modify_single_path(path)
            
            # Add metadata to all variations
            for i, variation in enumerate(path_variations):
                if i == 0:  # Original path
                    variation.update({
                        'algorithm': original_solution.get('algorithm', 'Original'),
                        'criteria': original_solution.get('criteria', {})
                    })
                else:  # Modified path
                    variation.update({
                        'algorithm': original_solution.get('algorithm', '') + ' + Modified',
                        'criteria': original_solution.get('criteria', {})
                    })
                    
            modified_solutions.extend(path_variations)
            
        return modified_solutions