import random
import time
import networkx as nx
from typing import List, Tuple, Dict, Set, Optional
from copy import deepcopy

class OFMOptimizer:
    def __init__(self, graph: nx.DiGraph):
        """Initialize the OFM optimizer with a transportation network graph."""
        self.graph = graph
        self.available_modes = {'bus', 'subway', 'trail'}
    
    def find_paths(self, 
                  origin_node: str, 
                  destination_node: str, 
                  criteria: Optional[Dict[str, bool]] = None,
                  num_filters: int = 10,
                  num_neighbors: int = 5,
                  iterations: int = 20,
                  R: float = 0.5) -> Tuple[List[Tuple[List[str], float, float, float, float]], float]:
        """
        Find optimized paths between nodes, returning ONLY non-comparable (Pareto-optimal) solutions.
        Ensures at least two different and non-comparable solutions are returned when possible.
            
        Returns:
            Tuple: (list of non-dominated paths with metrics, execution_time)
        """
        start_time = time.time()

        if criteria is None:
            criteria = {'distance': True, 'time': True, 'cost': True, 'quality': False}
            
        if not self._validate_nodes(origin_node, destination_node):
            return [], 0.0

        # Initialize
        filters = []
        pareto_front = set()
        
        # Generate initial solutions
        for _ in range(num_filters):
            path = self._random_walk(origin_node, destination_node)
            if path:
                metrics = self._calculate_path_metrics(path)
                solution = (tuple(path), *metrics)
                filters.append(solution)
                self._update_pareto_front(pareto_front, solution, criteria)
        
        # Optimization loop
        for _ in range(iterations):
            new_filters = []
            
            for current in filters:
                path, dist, t, cost, qual = current
                
                for _ in range(num_neighbors):
                    # Generate neighbor
                    if random.random() < R:
                        neighbor_path = self._random_walk(origin_node, destination_node)
                    else:
                        neighbor_path = self._change_transport_modes(deepcopy(list(path)))
                    
                    if neighbor_path:
                        neighbor_metrics = self._calculate_path_metrics(neighbor_path)
                        neighbor_sol = (tuple(neighbor_path), *neighbor_metrics)
                        
                        # Update Pareto front
                        if self._update_pareto_front(pareto_front, neighbor_sol, criteria):
                            new_filters.append(neighbor_sol)
            
            filters = new_filters if new_filters else filters
        
        # Convert to output format
        result = []
        for sol in pareto_front:
            path, *metrics = sol
            result.append((list(path), *metrics))
        
        # Ensure at least two different non-comparable solutions
        if len(result) < 2:
            result = self._find_additional_solutions(origin_node, destination_node, result, criteria)
        
        exec_time = time.time() - start_time
        return self._sort_paths(result, criteria)[:10], exec_time

    def _find_additional_solutions(self, 
                                origin: str, 
                                destination: str, 
                                existing_solutions: List[Tuple], 
                                criteria: Dict[str, bool]) -> List[Tuple]:
        """
        Secondary strategy to find additional solutions by removing edges from existing paths
        and searching for alternative paths in the modified graph.
        """
        if not existing_solutions:
            return existing_solutions
            
        # Collect all nodes from existing solutions
        all_nodes = set()
        for sol in existing_solutions:
            all_nodes.update(sol[0])
        
        # Create a copy of the original graph
        temp_graph = self.graph.copy()
        
        # Try removing different edges to find alternative paths
        max_attempts = 10
        attempts = 0
        new_solutions = existing_solutions.copy()
        
        while len(new_solutions) < 2 and attempts < max_attempts:
            attempts += 1
            
            # Select a random existing path
            base_path = random.choice(existing_solutions)[0]
            
            if len(base_path) < 2:
                continue
                
            # Remove a random edge from this path
            edge_to_remove = random.randint(0, len(base_path)-2)
            u, v = base_path[edge_to_remove], base_path[edge_to_remove+1]
            
            if temp_graph.has_edge(u, v):
                temp_graph.remove_edge(u, v)
                
                # Try to find alternative paths in the modified graph
                try:
                    # Find shortest path as an alternative (can be replaced with other methods)
                    alt_path = nx.shortest_path(temp_graph, origin, destination, weight='distance')
                    alt_metrics = self._calculate_path_metrics(alt_path)
                    alt_solution = (alt_path, *alt_metrics)
                    
                    # Check if it's non-comparable with existing solutions
                    is_non_dominated = True
                    for existing in new_solutions:
                        if self._dominates(existing, alt_solution, criteria) or \
                           self._dominates(alt_solution, existing, criteria):
                            is_non_dominated = False
                            break
                    
                    if is_non_dominated:
                        new_solutions.append(alt_solution)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
                
                # Restore the removed edge for next attempt
                temp_graph.add_edge(u, v, **self.graph[u][v])
        
        return new_solutions

    def _update_pareto_front(self, 
                           front: Set[Tuple], 
                           new_solution: Tuple, 
                           criteria: Dict[str, bool]) -> bool:
        """
        Helper: Update Pareto front with new solution.
        Returns True if solution was added to front.
        """
        to_remove = set()
        added = False
        
        # Check against existing solutions
        for existing in front:
            if self._dominates(existing, new_solution, criteria):
                return False  # New solution is dominated
            elif self._dominates(new_solution, existing, criteria):
                to_remove.add(existing)  # New solution dominates existing
        
        # Remove dominated solutions
        front.difference_update(to_remove)
        
        # Add new solution if not dominated
        if not any(self._dominates(existing, new_solution, criteria) for existing in front):
            front.add(new_solution)
            added = True
        
        return added

    # ALL ORIGINAL METHODS REMAIN UNCHANGED BELOW THIS POINT
    def _validate_nodes(self, origin: str, destination: str) -> bool:
        """Original implementation"""
        return (origin in self.graph.nodes and 
                destination in self.graph.nodes and 
                origin != destination)

    def _calculate_path_metrics(self, path: List[str]) -> Tuple[float, float, float, float]:
        """Original implementation"""
        if len(path) < 2:
            return (float('inf'), float('inf'), float('inf'), 0.0)
            
        distance = time_val = cost = 0.0
        mode_changes = 0
        previous_mode = None
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if self.graph.has_edge(u, v):
                edge = self.graph[u][v]
                distance += edge['distance']
                cost += edge.get('cost', edge['distance'] * 0.1)
                time_val += edge['time']
                
                current_mode = edge['mode']
                if previous_mode and current_mode != previous_mode:
                    mode_changes += 1
                previous_mode = current_mode
        
        quality = 1.0 / (1.0 + mode_changes)
        return (distance, time_val, cost, quality)

    def _dominates(self, solution1: Tuple, solution2: Tuple, criteria: Dict[str, bool]) -> bool:
        """Original implementation"""
        _, dist1, time1, cost1, qual1 = solution1
        _, dist2, time2, cost2, qual2 = solution2
        
        better_in_any = False
        
        if criteria.get('distance', False):
            if dist1 > dist2: return False
            if dist1 < dist2: better_in_any = True
                
        if criteria.get('time', False):
            if time1 > time2: return False
            if time1 < time2: better_in_any = True
                
        if criteria.get('cost', False):
            if cost1 > cost2: return False
            if cost1 < cost2: better_in_any = True
                
        if criteria.get('quality', False):
            if qual1 < qual2: return False
            if qual1 > qual2: better_in_any = True
                
        return better_in_any

    def _sort_paths(self, paths: List[Tuple], criteria: Dict[str, bool]) -> List[Tuple]:
        """Original implementation"""
        if not paths:
            return []
            
        weights = {
            'distance': 0.4 if criteria.get('distance', False) else 0,
            'time': 0.3 if criteria.get('time', False) else 0,
            'cost': 0.2 if criteria.get('cost', False) else 0,
            'quality': 0.1 if criteria.get('quality', False) else 0
        }
        
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            for key in weights:
                weights[key] /= weight_sum
        
        return sorted(paths, key=lambda x: (
            weights['distance'] * x[1] +
            weights['time'] * x[2] +
            weights['cost'] * x[3] -
            weights['quality'] * x[4]
        ))

    def _random_walk(self, origin: str, destination: str, max_steps: int = 100) -> Optional[List[str]]:
        """Original implementation"""
        path = [origin]
        current = origin
        visited = set([origin])
        steps = 0
        
        while current != destination and steps < max_steps:
            neighbors = list(self.graph.neighbors(current))
            unvisited = [n for n in neighbors if n not in visited]
            
            if unvisited:
                next_node = random.choice(unvisited)
            elif neighbors:
                next_node = random.choice(neighbors)
            else:
                break
                
            path.append(next_node)
            visited.add(next_node)
            current = next_node
            steps += 1
            
        return path if path[-1] == destination else None

    def _change_transport_modes(self, path: List[str]) -> Optional[List[str]]:
        """Original implementation"""
        if len(path) < 2:
            return None
            
        new_path = path.copy()
        
        for i in range(len(new_path) - 1):
            u, v = new_path[i], new_path[i+1]
            
            if not self.graph.has_edge(u, v):
                return None
                
            current_edge = self.graph[u][v]
            available_modes = self._get_available_modes(u, v)
            
            if available_modes:
                current_edge['mode'] = random.choice(list(available_modes))
        
        return new_path

    def _get_available_modes(self, u: str, v: str) -> Set[str]:
        """Original implementation"""
        if not self.graph.has_edge(u, v):
            return set()
            
        current_mode = self.graph[u][v]['mode']
        available = {current_mode}
        other_modes = list(self.available_modes - {current_mode})
        available.update(random.sample(other_modes, min(2, len(other_modes))))
        return available