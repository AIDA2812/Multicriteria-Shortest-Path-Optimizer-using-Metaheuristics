# aco.py
import random
import time
import networkx as nx
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import copy

class AntColonyOptimizer:
    def __init__(self, graph: nx.Graph):
        """Initialize the ACO optimizer with a transportation network graph"""
        self.graph = graph
        self.pheromone = {}  # Stores pheromone levels for edges
        self.parameters = {
            'alpha': 1.0,    # Pheromone importance
            'beta': 2.0,     # Heuristic importance
            'evaporation': 0.5,  # Evaporation rate
            'q0': 0.7        # Exploitation probability
        }
        self.initialize_pheromone()

    def initialize_pheromone(self):
        """Initialize pheromone levels on all edges"""
        for u, v in self.graph.edges():
            self.pheromone[(u, v)] = 1.0
            self.pheromone[(v, u)] = 1.0  # For undirected edges

    def find_shortest_paths(self, origin_node: str, destination_node: str, 
                          criteria: Dict[str, bool] = None,
                          n_ants: int = 10, 
                          n_iterations: int = 50,
                          alpha: float = 1.0,
                          beta: float = 2.0,
                          evaporation_rate: float = 0.5,
                          num_solutions: int = 5) -> Dict:
        """
        Find optimized paths using Ant Colony Optimization
        
        Args:
            origin_node: Start node ID
            destination_node: Target node ID
            criteria: Optimization criteria (cost/time/distance)
            n_ants: Number of ants per iteration
            n_iterations: Number of iterations
            alpha: Pheromone importance
            beta: Heuristic importance
            evaporation_rate: Pheromone evaporation rate
            num_solutions: Maximum number of non-dominated solutions to return
            
        Returns:
            Dictionary containing:
            - solutions: List of non-dominated path solutions
            - iterations: Number of iterations completed
            - execution_time: Total runtime
            - stats: Optimization statistics
            - criteria: Used optimization criteria
        """
        # Update algorithm parameters
        self.parameters.update({
            'alpha': alpha,
            'beta': beta,
            'evaporation': evaporation_rate
        })

        start_time = time.time()
        criteria = criteria or {'cost': True}  # Default to cost optimization
        
        # Validate nodes
        if origin_node not in self.graph or destination_node not in self.graph:
            return {
                'solutions': [],
                'iterations': 0,
                'execution_time': 0,
                'error': 'Invalid origin or destination node'
            }
            
        all_solutions = []  # Store all solutions found
        stats = {
            'best_cost': [],
            'best_time': [],
            'best_distance': [],
            'iteration_times': []
        }

        for iteration in range(n_iterations):
            iter_start = time.time()
            ant_paths = []
            ant_metrics = []
            
            # Generate solutions from all ants
            for _ in range(n_ants):
                path = self.construct_path(origin_node, destination_node, criteria)
                
                if path and path[-1] == destination_node:
                    metrics = self.calculate_path_metrics(path)
                    ant_paths.append(path)
                    ant_metrics.append(metrics)
                    all_solutions.append(metrics)  # Collect solution
            
            # Update pheromones if we found valid paths
            if ant_paths:
                self.update_pheromones(ant_paths, ant_metrics, criteria)
                
                # Find best in iteration by combined score for stats
                best_in_iter = min(ant_metrics, key=lambda x: self.calculate_solution_score(x))
                stats['best_cost'].append(best_in_iter['cost'])
                stats['best_time'].append(best_in_iter['time'])
                stats['best_distance'].append(best_in_iter['distance'])
            else:
                # No valid paths in iteration
                if iteration > 0:
                    # Carry forward previous best stats
                    stats['best_cost'].append(stats['best_cost'][-1])
                    stats['best_time'].append(stats['best_time'][-1])
                    stats['best_distance'].append(stats['best_distance'][-1])
                else:
                    # Use placeholders for first iteration
                    stats['best_cost'].append(float('inf'))
                    stats['best_time'].append(float('inf'))
                    stats['best_distance'].append(float('inf'))
            
            stats['iteration_times'].append(time.time() - iter_start)

        # Remove duplicate solutions
        unique_solutions = self.remove_duplicate_solutions(all_solutions)
        # Get non-dominated solutions
        active_objectives = [key for key, value in criteria.items() if value]
        non_dominated = self.get_non_dominated_solutions(unique_solutions, active_objectives)
        
        # Secondary strategy if fewer than two non-dominated solutions
        if len(non_dominated) < 2:
            # Collect all nodes from existing solutions
            all_nodes = set()
            for sol in unique_solutions:
                all_nodes.update(sol['path'])
                
            # Create a subgraph containing only these nodes
            subgraph = self.graph.subgraph(all_nodes).copy()
            
            # Create a temporary optimizer with the subgraph
            temp_optimizer = AntColonyOptimizer(subgraph)
            
            # Try to find alternative paths by removing edges
            additional_solutions = []
            attempts = 0
            max_attempts = min(10, len(unique_solutions) * 3)  # Limit attempts
            
            while len(additional_solutions) < 2 - len(non_dominated) and attempts < max_attempts:
                attempts += 1
                
                # Create a modified graph by removing some edges
                mod_graph = copy.deepcopy(subgraph)
                edges_to_remove = []
                
                # Remove edges from existing solutions
                for sol in unique_solutions:
                    path = sol['path']
                    if len(path) > 2:  # Need at least 2 edges to remove one
                        # Randomly select an edge to remove (avoid first/last for connectivity)
                        idx = random.randint(0, len(path)-2)
                        u, v = path[idx], path[idx+1]
                        if mod_graph.has_edge(u, v):
                            edges_to_remove.append((u, v))
                
                # Remove selected edges
                for u, v in edges_to_remove:
                    if mod_graph.has_edge(u, v):
                        mod_graph.remove_edge(u, v)
                
                # Run optimization on modified graph
                temp_optimizer.graph = mod_graph
                temp_optimizer.initialize_pheromone()
                
                # Run with reduced iterations for efficiency
                temp_result = temp_optimizer.find_shortest_paths(
                    origin_node, destination_node, criteria,
                    n_ants=5, n_iterations=5,
                    alpha=alpha, beta=beta,
                    evaporation_rate=evaporation_rate,
                    num_solutions=1
                )
                
                # Collect valid new solutions
                if temp_result['solutions']:
                    for sol in temp_result['solutions']:
                        path_str = ','.join(sol['path'])
                        # Check if solution is new
                        if all(path_str != ','.join(s['path']) for s in unique_solutions + additional_solutions):
                            additional_solutions.append(sol)
            
            # Add new solutions to the pool
            if additional_solutions:
                all_solutions += additional_solutions
                unique_solutions = self.remove_duplicate_solutions(all_solutions)
                non_dominated = self.get_non_dominated_solutions(unique_solutions, active_objectives)

        # Ensure at least two solutions (even if dominated)
        if len(non_dominated) < 2:
            # Sort all solutions by combined score
            unique_solutions.sort(key=lambda x: self.calculate_solution_score(x))
            # Add best dominated solutions to reach two
            for sol in unique_solutions:
                if sol not in non_dominated and len(non_dominated) < 2:
                    non_dominated.append(sol)

        # Return at least two non-dominated solutions if available
        if len(non_dominated) > num_solutions:
            non_dominated.sort(key=lambda x: self.calculate_solution_score(x))
            non_dominated = non_dominated[:num_solutions]

        return {
            'solutions': non_dominated,
            'iterations': n_iterations,
            'execution_time': time.time() - start_time,
            'stats': stats,
            'criteria': criteria
        }

    def adjust_path_modes(self, path: List[str], start_node: str, end_node: str, 
                         min_changes: int = 1, max_changes: int = 3) -> Dict:
        """
        Dynamically adjust transportation modes within an existing path.
        Attempts to modify between min_changes and max_changes segments.
        
        Args:
            path: The original path (list of nodes)
            start_node: Start node of the path
            end_node: End node of the path
            min_changes: Minimum number of mode changes to make (default 1)
            max_changes: Maximum number of mode changes to make (default 3)
            
        Returns:
            Dictionary containing:
            - original: The original path with metrics
            - adjusted: The adjusted path with new modes
            - changes: List of tuples describing changes made (u, v, old_mode, new_mode)
            - status: 'success' or 'error'
            - message: Descriptive message about the operation
        """
        if not path or path[0] != start_node or path[-1] != end_node:
            return {
                'status': 'error',
                'message': 'Invalid path or nodes',
                'original': None,
                'adjusted': None,
                'changes': []
            }
            
        # Calculate metrics for original path
        original_metrics = self.calculate_path_metrics(path)
        
        # Get all possible mode change points
        change_points = []
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            current_mode = self.graph[u][v]['mode']
            edge_modes = [data['mode'] for data in self.graph.get_edge_data(u, v).values()]
            
            # Only include if there are alternative modes available
            if len(edge_modes) > 1:
                change_points.append((i, u, v, current_mode, edge_modes))
        
        if not change_points:
            return {
                'status': 'error',
                'message': 'No segments with alternative modes available',
                'original': original_metrics,
                'adjusted': None,
                'changes': []
            }
        
        # Determine number of changes to make (between min and max)
        num_changes = min(max_changes, max(min_changes, len(change_points)))
        # Randomly select which segments to change
        selected_changes = random.sample(change_points, num_changes)
        
        # Create new mode sequence
        new_modes = []
        changes_made = []
        
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            current_mode = self.graph[u][v]['mode']
            
            # Check if this segment was selected for change
            change_segment = next((c for c in selected_changes if c[0] == i), None)
            
            if change_segment:
                # Get all alternative modes for this segment
                _, _, _, current_mode, edge_modes = change_segment
                other_modes = [m for m in edge_modes if m != current_mode]
                new_mode = random.choice(other_modes)
                new_modes.append(new_mode)
                changes_made.append((u, v, current_mode, new_mode))
            else:
                new_modes.append(current_mode)
        
        # Calculate metrics for adjusted path
        adjusted_metrics = self.calculate_path_metrics_for_modes(path.copy(), new_modes)
        
        return {
            'status': 'success',
            'message': f'Adjusted {num_changes} transport mode(s)',
            'original': original_metrics,
            'adjusted': adjusted_metrics,
            'changes': changes_made
        }

    def get_alternative_paths(self, path: List[str], start_node: str, end_node: str) -> Dict:
        """
        Find two alternative paths with different transport modes while maintaining the same route.
        
        Args:
            path: The original path (list of nodes)
            start_node: Start node of the path
            end_node: End node of the path
            
        Returns:
            Dictionary containing:
            - original: The original path with metrics
            - alternative1: First alternative path with different modes
            - alternative2: Second alternative path with different modes
            - status: 'success' or 'error'
        """
        if not path or path[0] != start_node or path[-1] != end_node:
            return {'status': 'error', 'message': 'Invalid path or nodes'}
            
        # Get all available transport modes in the network
        all_modes = set(data['mode'] for _, _, data in self.graph.edges(data=True))
        
        # Calculate metrics for original path
        original_metrics = self.calculate_path_metrics(path)
        
        alternatives = []
        attempts = 0
        max_attempts = 20
        
        while len(alternatives) < 2 and attempts < max_attempts:
            attempts += 1
            new_modes = []
            changes = []
            
            # Create a new mode sequence
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                current_mode = self.graph[u][v]['mode']
                
                # Get all possible modes for this edge
                edge_modes = [data['mode'] for data in self.graph.get_edge_data(u, v).values()]
                
                # Try to find a different mode
                if len(edge_modes) > 1:
                    # Filter out the current mode
                    other_modes = [m for m in edge_modes if m != current_mode]
                    if other_modes:
                        new_mode = random.choice(other_modes)
                        new_modes.append(new_mode)
                        changes.append((u, v, current_mode, new_mode))
                    else:
                        new_modes.append(current_mode)
                else:
                    new_modes.append(current_mode)
            
            # Only consider it if we actually made changes
            if changes:
                # Create the alternative path (same nodes, different modes)
                alt_path = path.copy()
                metrics = self.calculate_path_metrics_for_modes(alt_path, new_modes)
                
                # Ensure the alternative is meaningfully different
                if (metrics['modes'] != original_metrics['modes'] and 
                    not any(a['modes'] == metrics['modes'] for a in alternatives)):
                    alternatives.append({
                        'path': alt_path,
                        'modes': metrics['modes'],
                        'changes': changes,
                        'metrics': metrics
                    })
        
        result = {
            'original': original_metrics,
            'status': 'success',
            'message': f'Found {len(alternatives)} alternatives'
        }
        
        if len(alternatives) >= 1:
            result['alternative1'] = alternatives[0]
        if len(alternatives) >= 2:
            result['alternative2'] = alternatives[1]
            
        return result

    def calculate_path_metrics_for_modes(self, path: List[str], modes: List[str]) -> Dict:
        """
        Calculate metrics for a path with specific transport modes for each segment.
        
        Args:
            path: List of nodes in the path
            modes: List of transport modes for each segment
            
        Returns:
            Dictionary with path metrics
        """
        if len(path) - 1 != len(modes):
            raise ValueError("Number of modes must match number of path segments")
            
        cost = 0.0
        travel_time = 0.0
        distance = 0.0
        unique_modes = []
        
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            mode = modes[i]
            
            # Get the specific edge with this mode
            edge_data = None
            for key, data in self.graph.get_edge_data(u, v).items():
                if data['mode'] == mode:
                    edge_data = data
                    break
            
            if not edge_data:
                raise ValueError(f"No edge between {u} and {v} with mode {mode}")
                
            cost += edge_data.get('cost', edge_data['distance'] * 0.1)
            travel_time += edge_data.get('time', edge_data['distance'] * 2)
            distance += edge_data['distance']
            
            if not unique_modes or unique_modes[-1] != mode:
                unique_modes.append(mode)
                
            # Add transfer time if mode changes
            if i < len(path)-2 and modes[i+1] != mode:
                next_mode = modes[i+1]
                next_edge_data = None
                for key, data in self.graph.get_edge_data(v, path[i+2]).items():
                    if data['mode'] == next_mode:
                        next_edge_data = data
                        break
                
                if next_edge_data:
                    transfer_time = (edge_data.get('wait_time', 5) + 
                                   next_edge_data.get('wait_time', 5)) / 2
                    travel_time += transfer_time
        
        return {
            'path': path,
            'cost': cost,
            'time': travel_time,
            'distance': distance,
            'modes': unique_modes
        }

    def remove_duplicate_solutions(self, solutions: List[Dict]) -> List[Dict]:
        """Remove duplicate solutions based on path sequence"""
        seen = set()
        unique_solutions = []
        for sol in solutions:
            path_str = ','.join(sol['path'])
            if path_str not in seen:
                seen.add(path_str)
                unique_solutions.append(sol)
        return unique_solutions

    def get_non_dominated_solutions(self, solutions: List[Dict], objectives: List[str]) -> List[Dict]:
        """Identify non-dominated solutions (Pareto front) for given objectives"""
        if not solutions or not objectives:
            return solutions
            
        n = len(solutions)
        dominated = [False] * n
        
        # Mark dominated solutions
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                # Check if solution i dominates solution j
                if self.dominates(solutions[i], solutions[j], objectives):
                    dominated[j] = True
                # Check if solution j dominates solution i
                elif self.dominates(solutions[j], solutions[i], objectives):
                    dominated[i] = True
                    break  # Move to next i if i is dominated
                    
        # Collect non-dominated solutions
        non_dominated = []
        for i in range(n):
            if not dominated[i]:
                non_dominated.append(solutions[i])
                
        return non_dominated

    def dominates(self, a: Dict, b: Dict, objectives: List[str]) -> bool:
        """Check if solution A dominates solution B for given objectives"""
        at_least_as_good = True
        strictly_better = False
        
        for obj in objectives:
            # For minimization problems, lower values are better
            if a[obj] > b[obj]:
                at_least_as_good = False
                break
            elif a[obj] < b[obj]:
                strictly_better = True
                
        return at_least_as_good and strictly_better

    def construct_path(self, start_node: str, end_node: str, criteria: Dict[str, bool]) -> List[str]:
        """Construct a path from start to end node using pheromone trails"""
        path = [start_node]
        current_node = start_node
        visited = {start_node}
        
        while current_node != end_node:
            neighbors = [n for n in self.graph.neighbors(current_node) if n not in visited]
            
            if not neighbors:
                # Dead end - backtrack
                if len(path) > 1:
                    path.pop()
                    current_node = path[-1]
                    continue
                else:
                    return []  # No valid path found
                    
            # Calculate selection probabilities
            probabilities = []
            total = 0.0
            
            for neighbor in neighbors:
                tau = self.pheromone.get((current_node, neighbor), 1.0)
                eta = self.calculate_heuristic(current_node, neighbor, criteria)
                probabilities.append((neighbor, (tau ** self.parameters['alpha']) * 
                                   ((1.0/max(eta, 0.0001)) ** self.parameters['beta'])))
                total += probabilities[-1][1]
            
            # Normalize probabilities
            if total > 0:
                probabilities = [(n, p/total) for n, p in probabilities]
            else:
                probabilities = [(n, 1.0/len(neighbors)) for n in neighbors]
            
            # Sort by probability (descending)
            probabilities.sort(key=lambda x: -x[1])
            
            # Select next node
            if random.random() < self.parameters['q0']:
                next_node = probabilities[0][0]  # Exploitation
            else:
                next_node = self.roulette_wheel_select(probabilities)  # Exploration
            
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node
        
        return path

    def roulette_wheel_select(self, probabilities: List[Tuple[str, float]]) -> str:
        """Select a node using roulette wheel selection"""
        r = random.random()
        cumulative = 0.0
        for node, prob in probabilities:
            cumulative += prob
            if r <= cumulative:
                return node
        return probabilities[-1][0]  # Fallback

    def calculate_heuristic(self, u: str, v: str, criteria: Dict[str, bool]) -> float:
        """Calculate heuristic value for edge u-v based on criteria"""
        edge_data = self.graph[u][v]
        heuristic = 0.0
        total_weight = 0.0
        
        if criteria.get('cost', False):
            cost = edge_data.get('cost', edge_data['distance'] * 0.1)
            heuristic += cost * 0.5
            total_weight += 0.5
            
        if criteria.get('time', False):
            time_val = edge_data.get('time', edge_data['distance'] * 2)
            heuristic += time_val * 0.3
            total_weight += 0.3
            
        if criteria.get('distance', False):
            distance = edge_data['distance']
            heuristic += distance * 0.2
            total_weight += 0.2
            
        return heuristic / max(total_weight, 0.1)

    def update_pheromones(self, paths: List[List[str]], metrics: List[Dict], criteria: Dict[str, bool]):
        """Update pheromone levels based on ant paths"""
        # Evaporate existing pheromones
        for edge in self.pheromone:
            self.pheromone[edge] *= (1.0 - self.parameters['evaporation'])
            
        # Deposit new pheromones
        for path, metric in zip(paths, metrics):
            quality = 1.0 / self.calculate_solution_score(metric)
            delta_pheromone = quality * self.parameters['evaporation']
            
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                self.pheromone[(u, v)] += delta_pheromone
                self.pheromone[(v, u)] += delta_pheromone

    def calculate_solution_score(self, solution: Dict) -> float:
        """Calculate a combined quality score for a solution (lower is better)"""
        return (0.5 * solution['cost'] + 
                0.3 * solution['time'] + 
                0.2 * solution['distance'])

    def calculate_path_metrics(self, path: List[str]) -> Dict:
        """Calculate all metrics for a given path"""
        cost = 0.0
        travel_time = 0.0
        distance = 0.0
        modes = []
        
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            edge_data = self.graph[u][v]
            
            cost += edge_data.get('cost', edge_data['distance'] * 0.1)
            travel_time += edge_data.get('time', edge_data['distance'] * 2)
            distance += edge_data['distance']
            
            mode = edge_data['mode']
            if not modes or modes[-1] != mode:
                modes.append(mode)
                
            # Add transfer time if mode changes
            if i < len(path)-2:
                next_edge = self.graph[v][path[i+2]]
                if next_edge['mode'] != mode:
                    transfer_time = (edge_data.get('wait_time', 5) + 
                                   next_edge.get('wait_time', 5)) / 2
                    travel_time += transfer_time
        
        return {
            'path': path,
            'cost': cost,
            'time': travel_time,
            'distance': distance,
            'modes': modes
        }