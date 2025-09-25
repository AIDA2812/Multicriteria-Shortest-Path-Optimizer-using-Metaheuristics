# acs.py
import random
import time
import networkx as nx
import numpy as np
from typing import List, Dict, Optional, Tuple
from copy import deepcopy

class AntColonySystem:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.pheromone = {}
        self.heuristic = {}
        self.best_solutions = []
        self.initialize_pheromones()
        
        # ACS parameters
        self.alpha = 1.0  # Pheromone importance
        self.beta = 2.0   # Heuristic importance
        self.rho = 0.1    # Local evaporation rate
        self.xi = 0.1     # Global evaporation rate
        self.q0 = 0.9     # Exploitation probability
        self.min_pheromone = 0.01
        self.max_pheromone = 100.0
        
    def initialize_pheromones(self):
        """Initialize pheromone levels with safe defaults"""
        for u, v in self.graph.edges():
            self.pheromone[(u, v)] = 1.0
            self.pheromone[(v, u)] = 1.0
            
    def initialize_heuristics(self, criteria: Dict[str, bool]):
        """Initialize heuristic information with validation"""
        self.heuristic = {}
        default_value = 0.001
        
        for u, v, data in self.graph.edges(data=True):
            heuristic_value = default_value
            
            try:
                if criteria.get('cost', False):
                    heuristic_value += 1 / max(data.get('cost', 0.1), 0.001)
                if criteria.get('time', False):
                    heuristic_value += 1 / max(data.get('time', 0.1), 0.001)
                if criteria.get('distance', False):
                    heuristic_value += 1 / max(data.get('distance', 0.1), 0.001)
            except (TypeError, ZeroDivisionError):
                heuristic_value = default_value
            
            self.heuristic[(u, v)] = max(heuristic_value, default_value)
            self.heuristic[(v, u)] = max(heuristic_value, default_value)
    
    def find_shortest_paths(self, start_node: str, end_node: str, 
                          criteria: Dict[str, bool] = None,
                          n_ants: int = 10, 
                          n_iterations: int = 50,
                          alpha: float = 1.0,
                          beta: float = 2.0,
                          evaporation_rate: float = 0.1,
                          q0: float = 0.9,
                          rho: float = 0.1,
                          num_solutions: int = 5) -> Dict:
        """Main method to find paths with enhanced error handling and guaranteed minimum solutions"""
        try:
            # Set parameters
            self.alpha = float(alpha)
            self.beta = float(beta)
            self.xi = float(evaporation_rate)
            self.q0 = float(q0)
            self.rho = float(rho)
            num_solutions = int(num_solutions)
            
            # Validate criteria
            if not criteria or not isinstance(criteria, dict):
                criteria = {'cost': True}
            elif not any(criteria.values()):
                criteria = {'cost': True}
            
            self.initialize_heuristics(criteria)
            start_time = time.time()
            self.best_solutions = []
            all_solutions = []  # Collect all solutions found
            
            # Initialize with a valid path
            self.initialize_with_dijkstra(start_node, end_node, criteria)
            if self.best_solutions:
                all_solutions.append(self.best_solutions[0])
            
            # Main ACS loop
            for _ in range(n_iterations):
                ants_paths = []
                
                for _ in range(n_ants):
                    path = self.construct_solution(start_node, end_node, criteria)
                    if path and path['path'] and path['path'][-1] == end_node:
                        ants_paths.append(path)
                        all_solutions.append(path)  # Collect every solution
                
                if ants_paths:
                    self.update_pheromones(ants_paths)
            
            # Process all solutions to find non-dominated ones
            unique_solutions = self.remove_duplicate_solutions(all_solutions)
            active_objectives = [key for key, value in criteria.items() if value]
            non_dominated = self.get_non_dominated_solutions(unique_solutions, active_objectives)
            
            # If we don't have enough solutions, try the secondary strategy
            if len(non_dominated) < 2:
                additional_solutions = self.secondary_path_search(start_node, end_node, criteria, non_dominated)
                non_dominated.extend(additional_solutions)
                # Re-evaluate non-dominated solutions after adding new ones
                non_dominated = self.get_non_dominated_solutions(non_dominated, active_objectives)
            
            # Return up to num_solutions non-dominated solutions
            if len(non_dominated) > num_solutions:
                non_dominated = self.sort_by_combined_score(non_dominated)[:num_solutions]
            
            return {
                'solutions': non_dominated,
                'iterations': n_iterations,
                'execution_time': time.time() - start_time,
                'num_solutions': num_solutions,
                'criteria': criteria,
                'q0': self.q0,
                'rho': self.rho
            }
            
        except Exception as e:
            print(f"Error in ACS: {str(e)}")
            return {
                'solutions': [],
                'error': str(e)
            }
    
    def secondary_path_search(self, start_node: str, end_node: str, 
                           criteria: Dict[str, bool], 
                           existing_solutions: List[Dict]) -> List[Dict]:
        """
        Secondary strategy to find additional paths when initial search yields insufficient solutions.
        Retains nodes from existing paths and iteratively removes edges to find alternative paths.
        """
        additional_solutions = []
        
        # Collect all nodes from existing solutions
        all_nodes = set()
        for solution in existing_solutions:
            all_nodes.update(solution['path'])
        
        if not all_nodes:
            return []
        
        # Create a working copy of the graph
        working_graph = deepcopy(self.graph)
        
        # Try different edge removal strategies
        for strategy in ['random', 'high_cost', 'high_time', 'high_distance']:
            temp_graph = deepcopy(working_graph)
            
            # Remove edges based on strategy
            edges_to_remove = self.select_edges_to_remove(temp_graph, all_nodes, strategy)
            for u, v in edges_to_remove:
                if temp_graph.has_edge(u, v):
                    temp_graph.remove_edge(u, v)
            
            # Try to find new paths in the modified graph
            new_paths = self.find_paths_in_modified_graph(temp_graph, start_node, end_node, criteria)
            for path in new_paths:
                if not self.is_duplicate_path(path, existing_solutions + additional_solutions):
                    additional_solutions.append(path)
            
            # If we've found enough, stop early
            if len(additional_solutions) >= 2:
                break
        
        return additional_solutions[:2]  # Return at most 2 additional solutions

    def select_edges_to_remove(self, graph: nx.Graph, nodes_to_keep: set, strategy: str) -> List[Tuple]:
        """
        Select edges to remove based on the specified strategy.
        Returns list of (u, v) edges to remove.
        """
        edges = []
        
        if strategy == 'random':
            # Remove random edges between nodes we want to keep
            for u, v in graph.edges():
                if u in nodes_to_keep and v in nodes_to_keep:
                    edges.append((u, v))
            random.shuffle(edges)
            return edges[:min(3, len(edges))]  # Remove up to 3 edges
            
        elif strategy == 'high_cost':
            # Remove highest cost edges
            edge_list = []
            for u, v in graph.edges():
                if u in nodes_to_keep and v in nodes_to_keep:
                    cost = graph[u][v].get('cost', 0)
                    edge_list.append((u, v, cost))
            edge_list.sort(key=lambda x: x[2], reverse=True)
            return [(u, v) for u, v, _ in edge_list[:min(3, len(edge_list))]]
            
        elif strategy == 'high_time':
            # Remove edges with highest time
            edge_list = []
            for u, v in graph.edges():
                if u in nodes_to_keep and v in nodes_to_keep:
                    time_ = graph[u][v].get('time', 0)
                    edge_list.append((u, v, time_))
            edge_list.sort(key=lambda x: x[2], reverse=True)
            return [(u, v) for u, v, _ in edge_list[:min(3, len(edge_list))]]
            
        elif strategy == 'high_distance':
            # Remove edges with highest distance
            edge_list = []
            for u, v in graph.edges():
                if u in nodes_to_keep and v in nodes_to_keep:
                    distance = graph[u][v].get('distance', 0)
                    edge_list.append((u, v, distance))
            edge_list.sort(key=lambda x: x[2], reverse=True)
            return [(u, v) for u, v, _ in edge_list[:min(3, len(edge_list))]]
            
        return []

    def find_paths_in_modified_graph(self, graph: nx.Graph, start_node: str, 
                                   end_node: str, criteria: Dict[str, bool]) -> List[Dict]:
        """
        Find paths in a modified graph using simple path-finding algorithms.
        """
        paths = []
        
        # Try Dijkstra's algorithm with different weights
        for weight in ['cost', 'time', 'distance']:
            if criteria.get(weight, False):
                try:
                    path = nx.shortest_path(graph, start_node, end_node, weight=weight)
                    path_data = self._calculate_path_metrics_for_graph(graph, path)
                    if path_data:
                        paths.append(path_data)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
        
        # Try Yen's algorithm for k-shortest paths if we still don't have enough
        if len(paths) < 2:
            try:
                for path in nx.shortest_simple_paths(graph, start_node, end_node, weight='cost'):
                    path_data = self._calculate_path_metrics_for_graph(graph, path)
                    if path_data and not self.is_duplicate_path(path_data, paths):
                        paths.append(path_data)
                        if len(paths) >= 2:
                            break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
        
        return paths

    def _calculate_path_metrics_for_graph(self, graph: nx.Graph, path: List[str]) -> Optional[Dict]:
        """Calculate metrics for a given path in a specific graph"""
        if not path or len(path) < 2:
            return None
            
        try:
            cost = sum(graph[u][v].get('cost', 0) for u, v in zip(path[:-1], path[1:]))
            time_ = sum(graph[u][v]['time'] for u, v in zip(path[:-1], path[1:]))
            distance = sum(graph[u][v]['distance'] for u, v in zip(path[:-1], path[1:]))
            modes = []
            
            for u, v in zip(path[:-1], path[1:]):
                mode = graph[u][v]['mode']
                if not modes or mode != modes[-1]:
                    modes.append(mode)
            
            return {
                'path': path,
                'cost': cost,
                'travel_time': time_,
                'distance': distance,
                'modes': modes,
                'num_mode_changes': len(modes) - 1
            }
        except (KeyError, TypeError):
            return None

    def is_duplicate_path(self, new_path: Dict, existing_paths: List[Dict]) -> bool:
        """Check if a path is a duplicate of any existing paths"""
        new_path_tuple = tuple(new_path['path'])
        for existing in existing_paths:
            if tuple(existing['path']) == new_path_tuple:
                return True
        return False

    def remove_duplicate_solutions(self, solutions: List[Dict]) -> List[Dict]:
        """Remove duplicate solutions based on path sequence"""
        seen = set()
        unique = []
        for sol in solutions:
            path_tuple = tuple(sol['path'])
            if path_tuple not in seen:
                seen.add(path_tuple)
                unique.append(sol)
        return unique
    
    def get_non_dominated_solutions(self, solutions: List[Dict], objectives: List[str]) -> List[Dict]:
        """Identify non-dominated solutions (Pareto front) for given objectives"""
        if not solutions or not objectives:
            return solutions
        
        # Create key mapping for solution metrics
        key_map = {
            'cost': 'cost',
            'time': 'travel_time',
            'distance': 'distance'
        }
        actual_keys = [key_map[obj] for obj in objectives if obj in key_map]
        
        n = len(solutions)
        dominated = [False] * n
        
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if self.dominates(solutions[i], solutions[j], actual_keys):
                    dominated[j] = True
                elif self.dominates(solutions[j], solutions[i], actual_keys):
                    dominated[i] = True
                    break
                    
        return [solutions[i] for i in range(n) if not dominated[i]]
    
    def dominates(self, a: Dict, b: Dict, keys: List[str]) -> bool:
        """Check if solution A dominates solution B for given keys"""
        a_not_worse = True
        a_better = False
        
        for key in keys:
            a_val = a[key]
            b_val = b[key]
            if a_val > b_val:
                a_not_worse = False
            elif a_val < b_val:
                a_better = True
                
        return a_not_worse and a_better
    
    def sort_by_combined_score(self, solutions: List[Dict]) -> List[Dict]:
        """Sort solutions by combined quality score"""
        return sorted(solutions, key=lambda x: self.calculate_solution_quality(x), reverse=True)
    
    def initialize_with_dijkstra(self, start_node: str, end_node: str, criteria: Dict):
        """Initialize with a valid path using Dijkstra's algorithm"""
        try:
            weight = 'cost' if criteria.get('cost', False) else \
                    'time' if criteria.get('time', False) else \
                    'distance' if criteria.get('distance', False) else 'cost'
            
            path = nx.shortest_path(self.graph, start_node, end_node, weight=weight)
            
            # Calculate path metrics
            cost = sum(self.graph[u][v].get('cost', 0) for u, v in zip(path[:-1], path[1:]))
            time_ = sum(self.graph[u][v]['time'] for u, v in zip(path[:-1], path[1:]))
            distance = sum(self.graph[u][v]['distance'] for u, v in zip(path[:-1], path[1:]))
            modes = []
            
            for u, v in zip(path[:-1], path[1:]):
                mode = self.graph[u][v]['mode']
                if not modes or mode != modes[-1]:
                    modes.append(mode)
            
            solution = {
                'path': path,
                'cost': cost,
                'travel_time': time_,
                'distance': distance,
                'modes': modes,
                'criteria': criteria
            }
            
            self.best_solutions.append(solution)
            
            # Boost pheromone on this path
            for u, v in zip(path[:-1], path[1:]):
                self.pheromone[(u, v)] = 2.0
                self.pheromone[(v, u)] = 2.0
                
        except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
            pass
    
    def construct_solution(self, start_node: str, end_node: str, criteria: Dict) -> Optional[Dict]:
        """Build a path for a single ant with backtracking"""
        path = [start_node]
        visited = {start_node: True}
        metrics = {'cost': 0, 'travel_time': 0, 'distance': 0}
        modes = []
        max_steps = len(self.graph.nodes()) * 2  # Prevent infinite loops
        
        for _ in range(max_steps):
            current = path[-1]
            if current == end_node:
                break
                
            neighbors = [n for n in self.graph.neighbors(current) if n not in visited]
            if not neighbors:
                if len(path) > 1:  # Backtrack
                    path.pop()
                    continue
                return None  # No path found
            
            next_node = self.select_next_node(current, neighbors)
            edge = self.graph[current][next_node]
            
            # Update metrics
            metrics['cost'] += edge.get('cost', 0)
            metrics['travel_time'] += edge['time']
            metrics['distance'] += edge['distance']
            
            # Track modes
            if not modes or edge['mode'] != modes[-1]:
                modes.append(edge['mode'])
            
            # Local pheromone update
            self.pheromone[(current, next_node)] = max(
                (1 - self.rho) * self.pheromone.get((current, next_node), self.min_pheromone) + 
                self.rho * self.min_pheromone,
                self.min_pheromone
            )
            
            path.append(next_node)
            visited[next_node] = True
        
        if path[-1] != end_node:
            return None
            
        return {
            'path': path,
            'cost': metrics['cost'],
            'travel_time': metrics['travel_time'],
            'distance': metrics['distance'],
            'modes': modes,
            'criteria': criteria
        }
    
    def select_next_node(self, current: str, neighbors: List[str]) -> str:
        """Select next node with proper probability handling"""
        if not neighbors:
            return None
            
        if random.random() < self.q0:  # Exploitation
            best_node = None
            best_value = -1
            
            for node in neighbors:
                pheromone = self.pheromone.get((current, node), self.min_pheromone)
                heuristic = self.heuristic.get((current, node), 0.001)
                value = (pheromone ** self.alpha) * (heuristic ** self.beta)
                
                if value > best_value:
                    best_value = value
                    best_node = node
            
            return best_node
        else:  # Exploration
            probabilities = []
            total = 0.0
            
            for node in neighbors:
                pheromone = self.pheromone.get((current, node), self.min_pheromone)
                heuristic = self.heuristic.get((current, node), 0.001)
                value = (pheromone ** self.alpha) * (heuristic ** self.beta)
                probabilities.append(value)
                total += value
            
            # Normalize probabilities safely
            if total <= 0:
                return random.choice(neighbors)
                
            probabilities = [p / total for p in probabilities]
            
            # Select using numpy's random choice
            try:
                return neighbors[np.random.choice(len(neighbors), p=probabilities)]
            except:
                return random.choice(neighbors)
    
    def update_pheromones(self, solutions: List[Dict]):
        """Update pheromones with bounds checking"""
        # Global evaporation
        for edge in self.pheromone:
            self.pheromone[edge] = max(
                (1 - self.xi) * self.pheromone[edge],
                self.min_pheromone
            )
        
        # Add pheromone from solutions
        for solution in solutions:
            if not solution.get('path'):
                continue
                
            quality = self.calculate_solution_quality(solution)
            path = solution['path']
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                self.pheromone[(u, v)] = min(
                    max(
                        self.pheromone.get((u, v), self.min_pheromone) + quality,
                        self.min_pheromone
                    ),
                    self.max_pheromone
                )
    
    def calculate_solution_quality(self, solution: Dict) -> float:
        """Calculate pheromone deposit amount"""
        quality = 1.0
        criteria = solution.get('criteria', {})
        
        try:
            if criteria.get('cost', False):
                quality *= 1 / max(solution['cost'], 0.001)
            if criteria.get('time', False):
                quality *= 1 / max(solution['travel_time'], 0.001)
            if criteria.get('distance', False):
                quality *= 1 / max(solution['distance'], 0.001)
        except (KeyError, TypeError, ZeroDivisionError):
            quality = 1.0
            
        return quality

    def adjust_transport_modes(self, original_path: List[str], start_node: str, end_node: str, 
                            max_changes: int = 3) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Attempt to adjust transportation modes in a path while maintaining connectivity.
        Returns both original and modified paths with their metrics.
        """
        # Validate input
        if (not original_path or original_path[0] != start_node or original_path[-1] != end_node or
            max_changes < 1 or max_changes > 3):
            return None, None
            
        # Get original path data
        original_data = self._calculate_path_metrics(original_path)
        if not original_data:
            return None, None
            
        # Find potential edges for mode changes
        change_candidates = []
        for i in range(len(original_path) - 1):
            u, v = original_path[i], original_path[i+1]
            edge_data = self.graph.get_edge_data(u, v)
            if edge_data:
                # Find alternative edges between these nodes with different modes
                alternatives = []
                for _, _, data in self.graph.edges(u, data=True):
                    if data['v'] == v and data['mode'] != edge_data['mode']:
                        alternatives.append(data['mode'])
                if alternatives:
                    change_candidates.append((i, alternatives))
        
        if not change_candidates:
            return original_data, None
            
        # Try to make changes (1 to max_changes)
        modified_path = None
        for num_changes in range(1, min(max_changes, len(change_candidates)) + 1):
            # Try different combinations of changes
            for attempt in range(10):  # Limit attempts to prevent excessive computation
                changed_indices = random.sample(change_candidates, num_changes)
                temp_path = original_path.copy()
                changes_made = 0
                
                for idx, modes in changed_indices:
                    u, v = temp_path[idx], temp_path[idx+1]
                    new_mode = random.choice(modes)
                    
                    # Find an edge between these nodes with the new mode
                    for _, _, data in self.graph.edges(u, data=True):
                        if data['v'] == v and data['mode'] == new_mode:
                            # Update the path segment (though the nodes remain same)
                            changes_made += 1
                            break
                
                if changes_made == num_changes:
                    modified_data = self._calculate_path_metrics(temp_path)
                    if modified_data:
                        modified_path = modified_data
                        break
                
                if modified_path:
                    break
                    
            if modified_path:
                break
                
        return original_data, modified_path if modified_path else None
        
    def _calculate_path_metrics(self, path: List[str]) -> Optional[Dict]:
        """Calculate metrics for a given path"""
        if not path or len(path) < 2:
            return None
            
        try:
            cost = sum(self.graph[u][v].get('cost', 0) for u, v in zip(path[:-1], path[1:]))
            time_ = sum(self.graph[u][v]['time'] for u, v in zip(path[:-1], path[1:]))
            distance = sum(self.graph[u][v]['distance'] for u, v in zip(path[:-1], path[1:]))
            modes = []
            
            for u, v in zip(path[:-1], path[1:]):
                mode = self.graph[u][v]['mode']
                if not modes or mode != modes[-1]:
                    modes.append(mode)
            
            return {
                'path': path,
                'cost': cost,
                'travel_time': time_,
                'distance': distance,
                'modes': modes,
                'num_mode_changes': len(modes) - 1
            }
        except (KeyError, TypeError):
            return None