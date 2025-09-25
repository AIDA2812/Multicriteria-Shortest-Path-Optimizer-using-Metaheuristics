# dijkstra.py
import networkx as nx
import time

class DijkstraOptimizer:
    def __init__(self, graph):
        self.graph = graph
        
    def find_shortest_path(self, origin_node, destination_node, criteria=None):
        """Find shortest path using Dijkstra's algorithm between specific nodes"""
        if criteria is None:
            criteria = {'cost': True, 'time': False, 'distance': False}
        
        # Determine which weight to use based on criteria
        weight = None
        if criteria.get('cost', False):
            weight = 'cost'
        elif criteria.get('time', False):
            weight = 'time'
        elif criteria.get('distance', False):
            weight = 'distance'
        else:  # Default to cost
            weight = 'cost'
        
        try:
            path = nx.dijkstra_path(self.graph, origin_node, destination_node, weight=weight)
            return {
                "path": path,
                "valid": True
            }
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return {
                "path": [],
                "valid": False
            }

    def calculate_path_metrics(self, path):
        """Calculate all metrics for a path"""
        if not path or len(path) < 2:
            return {
                'cost': float('inf'),
                'time': float('inf'),
                'distance': float('inf')
            }
        
        cost = 0
        time = 0
        distance = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.graph.has_edge(u, v):
                edge_data = self.graph[u][v]
                cost += edge_data.get('cost', edge_data['distance'] * 0.1)
                time += edge_data['time']
                distance += edge_data['distance']
        
        return {
            'cost': cost,
            'time': time,
            'distance': distance
        }

    def get_path_modes(self, path):
        """Get the transportation modes used in a path"""
        if not path or len(path) < 2:
            return []
        
        modes = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.graph.has_edge(u, v):
                mode = self.graph[u][v]['mode']
                if not modes or modes[-1] != mode:
                    modes.append(mode)
        
        return modes