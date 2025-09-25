# optimizer.py
import networkx as nx
import matplotlib.pyplot as plt
import time
import random
from dijkstra import DijkstraOptimizer
from genetic_algorithm import GeneticAlgorithmOptimizer
from aco import AntColonyOptimizer
from acs import AntColonySystem  # Updated import
from ofm_algorithm import OFMOptimizer
from typing import Dict, List, Tuple, Optional

class MultiModalTransportationOptimizer:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes = []
        self.edges = []
        self.dijkstra_optimizer = None
        self.genetic_algorithm_optimizer = None
        self.aco_optimizer = None
        self.acs_optimizer = None  # Separate ACS optimizer
        self.ofm_optimizer = None
        
        self.transport_modes = {
            'bus': {'speed_range': (20, 40), 'wait_range': (5, 15)},
            'subway': {'speed_range': (30, 50), 'wait_range': (2, 10)},
            'train': {'speed_range': (60, 100), 'wait_range': (10, 20)},
            'walk': {'speed_range': (4, 6), 'wait_range': (0, 0)},
            'bike': {'speed_range': (10, 20), 'wait_range': (0, 5)},
            'taxi': {'speed_range': (30, 60), 'wait_range': (0, 5)},
            'plane': {'speed_range': (400, 600), 'wait_range': (30, 90)}
        }
        
    def load_data(self, nodes, edges):
        """Load transportation data into the optimizer"""
        self.nodes = nodes
        self.edges = edges
        
        G = nx.DiGraph()
        
        for node in nodes:
            G.add_node(
                node['id'],
                label=node['label'],
                type=node['type'],
                lat=node['lat'],
                lng=node['lng'],
                city=node['city']
            )
        
        for edge in edges:
            mode = edge['mode']
            speed_info = self.transport_modes.get(mode, {'speed_range': (30, 50), 'wait_range': (5, 10)})
            
            speed = random.uniform(*speed_info['speed_range'])
            wait_time = random.uniform(*speed_info['wait_range'])
            
            distance_km = edge['distance']
            travel_time = (distance_km / speed) * 60  # Convert to minutes
            
            G.add_edge(
                edge['source'],
                edge['target'],
                distance=distance_km,
                time=travel_time + wait_time,
                mode=mode,
                speed=speed,
                wait_time=wait_time,
                cost=edge.get('cost', distance_km * 0.1)
            )
        
        self.graph = G
        self.dijkstra_optimizer = DijkstraOptimizer(G)
        self.genetic_algorithm_optimizer = GeneticAlgorithmOptimizer(G)
        self.aco_optimizer = AntColonyOptimizer(G)
        self.acs_optimizer = AntColonySystem(G)  # Initialize ACS separately
        self.ofm_optimizer = OFMOptimizer(G)
        return G
    
    def dijkstra(self, origin_node, destination_node, criteria=None):
        """Dijkstra's algorithm implementation"""
        start_time = time.time()  # Start timing

        result = self.dijkstra_optimizer.find_shortest_path(origin_node, destination_node, criteria)

        if not result['valid']:
            return self._empty_result("Dijkstra")

        metrics = self.dijkstra_optimizer.calculate_path_metrics(result['path'])
        end_time = time.time()  # End timing

        return {
        "path": result['path'],
        "cost": metrics['cost'],
        "time": end_time - start_time,  # Execution time in seconds
        "travel_time": metrics['time'],
        "distance": metrics['distance'],
        "modes": self.dijkstra_optimizer.get_path_modes(result['path']),
        "algorithm": "Dijkstra",
        "criteria": criteria or {'cost': True}
        }
    def genetic_algorithm(self, origin_node, destination_node, criteria=None, 
                population_size=50, generations=100, early_stopping=10,
                num_solutions=5):
        """Genetic Algorithm implementation"""
    # Validate criteria first
        criteria = self._validate_criteria(criteria)
    
        result = self.genetic_algorithm_optimizer.find_path(
        origin_node, destination_node, criteria,
        population_size=population_size,
        generations=generations,
        early_stopping=early_stopping,
        num_solutions=num_solutions
        )
    
    # Ensure criteria is included in each solution
        for solution in result['solutions']:
            solution['criteria'] = criteria
        
        return {
        "solutions": result['solutions'],
        "execution_time": result['execution_time'],
        "num_generations": result['num_generations'],
        "criteria": criteria  # Include at top level too
        }
    def ant_colony_optimization(self, origin_node, destination_node, criteria=None, 
                          n_ants=10, n_iterations=50, alpha=1.0, beta=2.0,
                          evaporation_rate=0.5, num_solutions=5):
        """ACO implementation"""
        start_time = time.time()
        criteria = self._validate_criteria(criteria)

        result = self.aco_optimizer.find_shortest_paths(
        origin_node, destination_node,
        criteria=criteria,
        n_ants=n_ants,
        n_iterations=n_iterations,
        alpha=alpha,
        beta=beta,
        evaporation_rate=evaporation_rate,
        num_solutions=num_solutions
        )

    # Ensure criteria is included in the result
        result['criteria'] = criteria
        return self._format_aco_results(result, start_time, "Ant Colony Optimization")

    def ant_colony_system(self, origin_node, destination_node, criteria=None, 
                    n_ants=10, n_iterations=50, alpha=1.0, beta=2.0,
                    evaporation_rate=0.5, q0=0.9, rho=0.1, num_solutions=5):
        """ACS implementation"""
        start_time = time.time()
        criteria = self._validate_criteria(criteria)

        result = self.acs_optimizer.find_shortest_paths(
        origin_node, destination_node,
        criteria=criteria,
        n_ants=n_ants,
        n_iterations=n_iterations,
        alpha=alpha,
        beta=beta,
        evaporation_rate=evaporation_rate,
        q0=q0,
        rho=rho,
        num_solutions=num_solutions
        )

    # Ensure criteria is included in the result
        result['criteria'] = criteria
        return self._format_aco_results(result, start_time, "Ant Colony System")
    def ofm_algorithm(self, origin_node, destination_node, criteria=None, 
                num_filters=10, num_neighbors=5, iterations=20, R=0.5):
        """OFM algorithm implementation"""
        criteria = self._validate_criteria(criteria, ofm=True)
     
        paths, exec_time = self.ofm_optimizer.find_paths(  # Unpack the returned tuple
        origin_node, destination_node,
        criteria=criteria,
        num_filters=num_filters,
        num_neighbors=num_neighbors,
        iterations=iterations,
        R=R
    )
    
        return {
        "paths": paths,
        "time": exec_time,  # Use the measured execution time
        "algorithm": "OFM",
        "criteria": criteria
        }


    def _format_ga_results(self, result, start_time, criteria):
        """Format GA results consistently"""
        exec_time = time.time() - start_time
        solutions = []
        
        for sol in result.get('solutions', []):
            solutions.append({
                "path": sol.get('path', []),
                "cost": sol.get('cost', 0),
                "time": exec_time,
                "travel_time": sol.get('travel_time', 0),
                "distance": sol.get('distance', 0),
                "modes": sol.get('modes', []),
                "fitness": sol.get('fitness', 0),
                "algorithm": "Genetic Algorithm",
                "criteria": criteria
            })
            
        return {
            "solutions": solutions,
            "execution_time": exec_time,
            "iterations": result.get('generations', 0),
            "stats": result.get('stats', [])
        }

    def _format_aco_results(self, result, start_time, algorithm_name):
        """Format ACO/ACS results consistently with validation"""
        exec_time = result.get('execution_time', time.time() - start_time)
        solutions = []
    
    # Get criteria from the result if available, otherwise use empty dict
        result_criteria = result.get('criteria', {})
    
        for sol in result.get('solutions', []):
          # Validate path exists and is non-empty
            if not sol.get('path') or len(sol['path']) < 2:
                continue

        # Use the solution's criteria if available, otherwise fall back to result criteria
            solution_criteria = sol.get('criteria', result_criteria)
        
            solutions.append({
            "path": sol['path'],
            "cost": sol.get('cost', self._calculate_path_cost(sol['path'])),
            "time": sol.get('time', self._calculate_path_time(sol['path'])),
            "distance": sol.get('distance', self._calculate_path_distance(sol['path'])),
            "modes": self._get_path_modes(sol['path']),
            "algorithm": algorithm_name,
            "criteria": solution_criteria,  # Use the determined criteria
            "stats": result.get('stats', [])
            })
   
    # Ensure we have the requested number of solutions
        num_solutions = result.get('num_solutions', 5)
        valid_solutions = [s for s in solutions if s['path']]
        while len(valid_solutions) < num_solutions:
            valid_solutions.append(self._empty_result(algorithm_name))

        return {
        "solutions": valid_solutions[:num_solutions],
        "execution_time": exec_time,
        "iterations": result.get('iterations', 0),
        "stats": result.get('stats', []),
        "criteria": result_criteria  # Include criteria in the final result
        }
        
    def _validate_criteria(self, criteria, ofm=False):
        """Ensure valid criteria configuration"""
        if criteria is None:
            return {'cost': True} if not ofm else {'cost': True, 'distance': True}
    
    # Convert to dict if it's not already
        if not isinstance(criteria, dict):
            return {'cost': True}
    
    # Ensure we have all expected criteria keys
        expected_criteria = ['cost', 'time', 'distance']
        valid_criteria = {}
    
        for crit in expected_criteria:
            valid_criteria[crit] = bool(criteria.get(crit, False))
    
    # Ensure at least one criterion is selected
        if not any(valid_criteria.values()):
            valid_criteria['cost'] = True
    
        return valid_criteria
    def _empty_result(self, algorithm):
        """Return empty result structure"""
        return {
            "path": [],
            "cost": float('inf'),
            "time": 0,
            "travel_time": float('inf'),
            "distance": float('inf'),
            "modes": [],
            "algorithm": algorithm,
            "criteria": {}
        }


    def get_edge_details(self, u, v):
        """Get detailed information about an edge"""
        if not self.graph.has_edge(u, v):
            return None
            
        edge = self.graph[u][v]
        return {
            'source': u,
            'target': v,
            'distance': edge['distance'],
            'time': edge['time'],
            'travel_time': edge['time'] - edge['wait_time'],
            'wait_time': edge['wait_time'],
            'speed': edge['speed'],
            'mode': edge['mode'],
            'cost': edge.get('cost', 0)
        }

    def get_path_details(self, path):
        """Get detailed information about a complete path"""
        if not path or len(path) < 2:
            return None
            
        details = {
            'total_distance': self._calculate_path_distance(path),
            'total_time': self._calculate_path_time(path),
            'total_cost': self._calculate_path_cost(path),
            'modes': self._get_path_modes(path),
            'segments': []
        }
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.graph.has_edge(u, v):
                edge_details = self.get_edge_details(u, v)
                details['segments'].append(edge_details)
                
                # Add transfer time if mode changes
                if i < len(path) - 2:
                    next_edge = self.get_edge_details(v, path[i+2])
                    if next_edge and edge_details['mode'] != next_edge['mode']:
                        transfer_time = (edge_details['wait_time'] + next_edge['wait_time']) / 2
                        details['segments'].append({
                            'source': v,
                            'target': v,
                            'description': f"Transfer from {edge_details['mode']} to {next_edge['mode']}",
                            'time': transfer_time,
                            'is_transfer': True
                        })
        
        return details

    def _calculate_path_cost(self, path):
        """Calculate the total cost of a path"""
        if not path or len(path) < 2:
            return float('inf')
        
        cost = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.graph.has_edge(u, v):
                cost += self.graph[u][v].get('cost', self.graph[u][v]['distance'] * 0.1)
        
        return cost
    
    def _calculate_path_distance(self, path):
        """Calculate the total distance of a path"""
        if not path or len(path) < 2:
            return float('inf')
        
        distance = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.graph.has_edge(u, v):
                distance += self.graph[u][v]['distance']
        
        return distance
    
    def _calculate_path_time(self, path):
        """Calculate the total time of a path including travel and waiting times"""
        if not path or len(path) < 2:
            return float('inf')
        
        total_time = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.graph.has_edge(u, v):
                total_time += self.graph[u][v]['time']
                
                # Add transfer waiting time if mode changes
                if i < len(path) - 2:
                    next_edge = self.graph[v][path[i+2]]
                    if next_edge['mode'] != self.graph[u][v]['mode']:
                        transfer_time = (self.graph[u][v]['wait_time'] + next_edge['wait_time']) / 2
                        total_time += transfer_time
        
        return total_time
    
    def _get_path_modes(self, path):
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

    def visualize_path(self, path, save_path=None):
        """Visualize a specific path with detailed time information"""
        if not path:
            print("No path to visualize")
            return
            
        path_details = self.get_path_details(path)
        if not path_details:
            return
            
        plt.figure(figsize=(15, 10))
        
        # Create position mapping
        pos = {}
        node_colors = []
        node_sizes = []
        
        for node_id in self.graph.nodes():
            node = self.graph.nodes[node_id]
            pos[node_id] = (node['lng'], node['lat'])
            node_colors.append('red' if node_id in path else 'lightgray')
            node_sizes.append(150 if node_id in path else 30)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7)
        
        # Draw all edges in light gray
        nx.draw_networkx_edges(self.graph, pos, edge_color='lightgray', width=0.5, alpha=0.3)
        
        # Draw path edges with different colors
        edge_colors = {
            'bus': 'green',
            'train': 'blue',
            'subway': 'orange',
            'walk': 'gray',
            'bike': 'cyan',
            'taxi': 'yellow',
            'plane': 'red'
        }
        
        path_edges = list(zip(path[:-1], path[1:]))
        for u, v in path_edges:
            mode = self.graph[u][v]['mode']
            nx.draw_networkx_edges(self.graph, pos, edgelist=[(u, v)], 
                                 edge_color=edge_colors.get(mode, 'black'),
                                 width=3.0, alpha=1.0)
        
        # Draw labels
        labels = {node_id: node_id for node_id in path}
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=8)
        
        # Create detailed information text
        info_text = f"Total Distance: {path_details['total_distance']:.2f} km\n"
        info_text += f"Total Time: {path_details['total_time']:.2f} min\n"
        info_text += f"Total Cost: ${path_details['total_cost']:.2f}\n"
        info_text += f"Transport Modes: {', '.join(path_details['modes'])}\n\n"
        info_text += "Path Details:\n"
        
        for segment in path_details['segments']:
            if segment.get('is_transfer', False):
                info_text += (f"Transfer at {segment['source']}: {segment['description']} "
                             f"(Wait: {segment['time']:.1f} min)\n")
            else:
                info_text += (f"{segment['source']} → {segment['target']}: "
                             f"{segment['distance']:.2f} km, "
                             f"Time: {segment['time']:.1f} min "
                             f"(Travel: {segment['travel_time']:.1f} min + Wait: {segment['wait_time']:.1f} min), "
                             f"Speed: {segment['speed']:.1f} km/h, "
                             f"Mode: {segment['mode']}\n")
        
        plt.title(f"Optimal Path from {path[0]} to {path[-1]}")
        plt.figtext(0.5, 0.01, info_text, ha='center', va='bottom', fontsize=9)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def get_cities(self):
        """Get list of unique cities in the network"""
        cities = set()
        for node in self.nodes:
            cities.add(node['city'])
        return sorted(cities)

    def compare_algorithms(self, origin_city, destination_city):
        """Compare different algorithms for city-to-city routing"""
        # Get all nodes in the origin and destination cities
        origin_nodes = [node['id'] for node in self.nodes if node['city'] == origin_city]
        dest_nodes = [node['id'] for node in self.nodes if node['city'] == destination_city]
        
        if not origin_nodes or not dest_nodes:
            return {}
        
        # Select representative nodes (first node in each city)
        origin_node = origin_nodes[0]
        dest_node = dest_nodes[0]
        
        # Run each algorithm with default parameters
        results = {
            'dijkstra': self.dijkstra(origin_node, dest_node),
            'genetic_algorithm': self.genetic_algorithm(origin_node, dest_node),
            'ant_colony_optimization': self.ant_colony_optimization(origin_node, dest_node),
            'ant_colony_system': self.ant_colony_system(origin_node, dest_node),
            'ofm_algorithm': self.ofm_algorithm(origin_node, dest_node)
        }
        
        return results