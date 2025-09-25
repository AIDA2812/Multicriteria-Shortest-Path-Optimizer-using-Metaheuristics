# genetic_algorithm.py
import random
import numpy as np
import networkx as nx
from typing import List, Dict, Optional, Tuple
import time 

class GeneticAlgorithmOptimizer:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.best_fitness_history = []
        
    def find_path(self, origin_node: str, destination_node: str, criteria: Optional[Dict[str, bool]] = None, 
             population_size: int = 50, generations: int = 100, 
             early_stopping: int = 10, elitism_ratio: float = 0.1,
             initial_mutation_rate: float = 0.2, min_mutation_rate: float = 0.05,
             initial_crossover_rate: float = 0.8, num_solutions: int = 3) -> Dict:
        """Find optimal path using Genetic Algorithm with improved features"""
        start_time = time.time()

        if criteria is None:
            criteria = {'cost': True, 'time': True, 'distance': True}
        
        # Ensure at least one criterion is selected
        if not any(criteria.values()):
            criteria['cost'] = True
            print("Warning: No criteria selected. Defaulting to cost optimization.")
    
        random.seed()
        mutation_rate = initial_mutation_rate
        self.best_fitness_history = []
        self.all_solutions_history = []
    
        # Initialize population
        population = self._initialize_population(origin_node, destination_node, population_size)
    
        best_individual = None
        best_fitness = float('inf')
        no_improvement_count = 0
    
        # Evolution loop
        for generation in range(generations):
            current_solutions = []
            fitness_scores = []
        
            for path in population:
                metrics = self._calculate_path_metrics(path)
                fitness = self._evaluate_path_fitness(path, criteria)
            
                solution = {
                    'path': path,
                    'cost': metrics['cost'],
                    'time': metrics['time'],
                    'travel_time': metrics['time'],
                    'distance': metrics['distance'],
                    'modes': self._get_path_modes(path),
                    'fitness': fitness,
                    'criteria': criteria
                }
             
                current_solutions.append(solution)
                fitness_scores.append(fitness)
        
            # Store all solutions from this generation
            self.all_solutions_history.append(current_solutions)
        
            # Track best fitness
            min_fitness_idx = np.argmin(fitness_scores)
            current_best_fitness = fitness_scores[min_fitness_idx]
            self.best_fitness_history.append(current_best_fitness)
        
            # Update best individual
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[min_fitness_idx]
                no_improvement_count = 0
                mutation_rate = max(min_mutation_rate, mutation_rate * 0.9)
            else:
                no_improvement_count += 1
                mutation_rate = min(0.5, mutation_rate * 1.1)
        
            # Early stopping condition
            if no_improvement_count >= early_stopping:
                break
        
            # Elitism - select top individuals to survive
            elite_size = max(1, int(population_size * elitism_ratio))
            elite_indices = np.argpartition(fitness_scores, elite_size)[:elite_size]
            elites = [population[i] for i in elite_indices]
        
            # Selection - tournament selection for parents
            selected_parents = self._tournament_selection(population, fitness_scores, 
                                                    population_size - elite_size)
        
            # Crossover - produce offspring
            offspring = self._crossover(selected_parents, origin_node, destination_node, 
                                  crossover_rate=initial_crossover_rate)
        
            # Mutation - introduce variations
            mutated_offspring = self._mutation(offspring, origin_node, destination_node, 
                                         mutation_rate=mutation_rate)
        
            # Create new generation (elites + offspring)
            population = elites + mutated_offspring
        
            # Maintain population size
            population = population[:population_size]
    
        # Get all unique solutions from all generations
        unique_solutions = {}
        for gen_solutions in self.all_solutions_history:
            for sol in gen_solutions:
                path_str = ','.join(sol['path'])
                if path_str not in unique_solutions or sol['fitness'] < unique_solutions[path_str]['fitness']:
                    unique_solutions[path_str] = sol
    
        # Get non-dominated solutions
        non_dominated = self._get_non_dominated_solutions(list(unique_solutions.values()))
        
        # If no valid solutions found, return empty result
        if not non_dominated or non_dominated[0]['fitness'] == float('inf'):
            return {
                'solutions': [self._empty_result()],
                'best_fitness_history': self.best_fitness_history,
                'num_generations': generation + 1,
                'execution_time': time.time() - start_time
            }
    
        return {
            'solutions': non_dominated[:num_solutions],
            'best_fitness_history': self.best_fitness_history,
            'num_generations': generation + 1,
            'criteria': criteria,                      
            'execution_time': time.time() - start_time
        }

    def _get_non_dominated_solutions(self, solutions: List[Dict]) -> List[Dict]:
        """Identify non-dominated (Pareto optimal) solutions"""
        if not solutions:
            return []
            
        # Extract objectives based on active criteria
        active_criteria = [k for k, v in solutions[0]['criteria'].items() if v]
        if not active_criteria:
            active_criteria = ['cost']  # Default
        
        # Filter out invalid solutions
        valid_solutions = [s for s in solutions if s['fitness'] < float('inf')]
        if not valid_solutions:
            return []
            
        # Initialize non-dominated set with first solution
        non_dominated = [valid_solutions[0]]
        
        for candidate in valid_solutions[1:]:
            is_dominated = False
            to_remove = []
            
            # Check against all current non-dominated solutions
            for i, nd_solution in enumerate(non_dominated):
                # Count how many objectives the candidate is worse or equal in
                worse_or_equal = 0
                better = 0
                
                for criterion in active_criteria:
                    if candidate[criterion] > nd_solution[criterion]:
                        worse_or_equal += 1
                    elif candidate[criterion] < nd_solution[criterion]:
                        better += 1
                
                # If candidate is worse in all objectives, it's dominated
                if worse_or_equal == len(active_criteria) and better == 0:
                    is_dominated = True
                    break
                
                # If existing solution is dominated by candidate, mark for removal
                if better == len(active_criteria) and worse_or_equal == 0:
                    to_remove.append(i)
            
            # Remove solutions that are dominated by candidate
            for i in sorted(to_remove, reverse=True):
                non_dominated.pop(i)
            
            # Add candidate if it's not dominated
            if not is_dominated:
                non_dominated.append(candidate)
        
        return non_dominated

    def _empty_result(self) -> Dict:
        """Return empty result structure with all required keys"""
        return {
            "path": [], 
            "cost": float('inf'), 
            "time": float('inf'),
            "travel_time": float('inf'),
            "distance": float('inf'),
            "modes": [],
            "fitness": float('inf')
        }

    def _evaluate_path_fitness(self, path: List[str], criteria: Dict[str, bool]) -> float:
        """Evaluate fitness of a path (lower is better) with normalized criteria"""
        if not path or len(path) < 2:
            return float('inf')
        
        # Check path validity
        if not self._is_valid_path(path):
            return float('inf')
        
        metrics = self._calculate_path_metrics(path)
        
        # Calculate weighted fitness with normalization
        fitness = 0
        total_weight = 0
        
        # Define max values for normalization
        max_values = {
            'cost': 1000,    # Maximum expected cost ($)
            'time': 500,     # Maximum expected time (minutes)
            'distance': 1000 # Maximum expected distance (km)
        }
        
        # Calculate weights based on selected criteria
        weights = {k: 1.0 if v else 0.0 for k, v in criteria.items()}
        total_weight = sum(weights.values())
        
        # If no criteria selected (shouldn't happen due to check in find_path), default to cost
        if total_weight == 0:
            weights['cost'] = 1.0
            total_weight = 1.0
        
        # Normalize and weight each selected criterion
        if weights.get('cost', 0) > 0:
            normalized_cost = min(metrics['cost'] / max_values['cost'], 1.0)
            fitness += normalized_cost * (weights['cost'] / total_weight)
        
        if weights.get('time', 0) > 0:
            normalized_time = min(metrics['time'] / max_values['time'], 1.0)
            fitness += normalized_time * (weights['time'] / total_weight)
        
        if weights.get('distance', 0) > 0:
            normalized_dist = min(metrics['distance'] / max_values['distance'], 1.0)
            fitness += normalized_dist * (weights['distance'] / total_weight)
        
        # Penalize long paths to prefer shorter solutions
        length_penalty = len(path) / 20  # Penalize paths longer than 20 nodes
        return fitness * (1 + length_penalty)

    def _initialize_population(self, origin: str, destination: str, population_size: int) -> List[List[str]]:
        """Initialize diverse population using multiple methods"""
        population = []
        
        # Method 1: Random walks (50%)
        for _ in range(population_size // 2):
            path = self._random_walk(origin, destination)
            if path:
                population.append(path)
        
        # Method 2: Dijkstra with random weights (30%)
        for _ in range(int(population_size * 0.3)):
            try:
                temp_graph = self.graph.copy()
                for u, v in temp_graph.edges():
                    # Randomize weights to get different paths
                    temp_graph[u][v]['weight'] = temp_graph[u][v]['distance'] * random.uniform(0.5, 2.0)
                
                path = nx.dijkstra_path(temp_graph, origin, destination, weight='weight')
                population.append(path)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
        
        # Method 3: Greedy paths (20%)
        for _ in range(int(population_size * 0.2)):
            path = self._greedy_path(origin, destination)
            if path:
                population.append(path)
        
        # Fill remaining with random walks if needed
        while len(population) < population_size:
            path = self._random_walk(origin, destination)
            if path:
                population.append(path)
        
        return population

    def _random_walk(self, origin: str, destination: str, max_steps: int = 100) -> Optional[List[str]]:
        """Generate random walk with restart mechanism"""
        path = [origin]
        current = origin
        visited = set([origin])
        
        for _ in range(max_steps):
            neighbors = list(self.graph.neighbors(current))
            
            if not neighbors:
                break  # Dead end
            
            # Prefer unvisited nodes
            unvisited = [n for n in neighbors if n not in visited]
            next_node = random.choice(unvisited) if unvisited else random.choice(neighbors)
            
            path.append(next_node)
            visited.add(next_node)
            current = next_node
            
            if current == destination:
                return path
            
            # Occasionally restart from origin to explore different paths
            if random.random() < 0.05:
                path.append(origin)
                current = origin
        
        return path if path[-1] == destination else None

    def _greedy_path(self, origin: str, destination: str) -> Optional[List[str]]:
        """Generate path using greedy approach towards destination"""
        path = [origin]
        current = origin
        
        for _ in range(100):  # Prevent infinite loops
            if current == destination:
                return path
                
            neighbors = list(self.graph.neighbors(current))
            if not neighbors:
                break
                
            # Choose neighbor closest to destination (if position data available)
            try:
                # Try to use geographical position if available
                dest_pos = (self.graph.nodes[destination]['lng'], self.graph.nodes[destination]['lat'])
                current_pos = (self.graph.nodes[current]['lng'], self.graph.nodes[current]['lat'])
                
                def distance_to_dest(node):
                    node_pos = (self.graph.nodes[node]['lng'], self.graph.nodes[node]['lat'])
                    return ((node_pos[0]-dest_pos[0])**2 + (node_pos[1]-dest_pos[1])**2)**0.5
                
                next_node = min(neighbors, key=distance_to_dest)
            except KeyError:
                # Fallback to random selection if no position data
                next_node = random.choice(neighbors)
            
            path.append(next_node)
            current = next_node
        
        return path if path[-1] == destination else None

    def _tournament_selection(self, population: List[List[str]], fitness_scores: List[float], 
                            num_parents: int, tournament_size: int = 3) -> List[List[str]]:
        """Select parents using tournament selection"""
        selected = []
        
        for _ in range(num_parents):
            # Randomly select tournament participants
            tournament_indices = random.sample(range(len(population)), 
                                min(tournament_size, len(population)))
            # Select the best from the tournament
            winner_idx = min(tournament_indices, key=lambda i: fitness_scores[i])
            selected.append(population[winner_idx])
        
        return selected

    def _crossover(self, parents: List[List[str]], origin: str, destination: str, 
                  crossover_rate: float = 0.8) -> List[List[str]]:
        """Ordered crossover for paths"""
        offspring = []
        
        # Pair up parents
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i+1]
            
            if random.random() > crossover_rate or len(parent1) < 3 or len(parent2) < 3:
                offspring.extend([parent1, parent2])
                continue
                
            # Find common nodes (excluding origin and destination)
            common_nodes = set(parent1[1:-1]) & set(parent2[1:-1])
            
            if not common_nodes:
                offspring.extend([parent1, parent2])
                continue
                
            # Select crossover point from common nodes
            crossover_point = random.choice(list(common_nodes))
            
            # Get crossover indices
            idx1 = parent1.index(crossover_point)
            idx2 = parent2.index(crossover_point)
            
            # Create offspring
            child1 = parent1[:idx1] + parent2[idx2:]
            child2 = parent2[:idx2] + parent1[idx1:]
            
            # Repair duplicates in offspring
            child1 = self._repair_path(child1, origin, destination)
            child2 = self._repair_path(child2, origin, destination)
            
            offspring.extend([child1, child2])
        
        return offspring

    def _repair_path(self, path: List[str], origin: str, destination: str) -> List[str]:
        """Repair path by removing duplicates and ensuring connectivity"""
        if not path:
            return []
            
        # Remove duplicates while preserving order
        seen = set()
        repaired = []
        for node in path:
            if node not in seen:
                seen.add(node)
                repaired.append(node)
        
        # Ensure starts and ends correctly
        if not repaired or repaired[0] != origin:
            repaired.insert(0, origin)
        if not repaired or repaired[-1] != destination:
            repaired.append(destination)
        
        # Ensure path is valid
        valid_path = [repaired[0]]
        for i in range(1, len(repaired)):
            if self.graph.has_edge(valid_path[-1], repaired[i]):
                valid_path.append(repaired[i])
            else:
                # Find shortest path between disconnected nodes
                try:
                    subpath = nx.shortest_path(self.graph, valid_path[-1], repaired[i])
                    valid_path.extend(subpath[1:])
                except nx.NetworkXNoPath:
                    # If no path, truncate at last valid node
                    break
        
        return valid_path

    def _mutation(self, population: List[List[str]], origin: str, destination: str, 
                 mutation_rate: float = 0.2) -> List[List[str]]:
        """Mutate paths using several mutation operators"""
        mutated = []
        
        for path in population:
            if random.random() > mutation_rate or len(path) < 3:
                mutated.append(path)
                continue
                
            # Randomly select mutation operator
            operator = random.choice([
                self._point_mutation,
                self._segment_mutation,
                self._scramble_mutation
            ])
            
            mutated_path = operator(path, origin, destination)
            mutated.append(mutated_path if mutated_path else path)
        
        return mutated

    def _point_mutation(self, path: List[str], origin: str, destination: str) -> Optional[List[str]]:
        """Replace a single node in the path"""
        if len(path) <= 2:
            return None
            
        # Select mutation point (not origin or destination)
        mut_point = random.randint(1, len(path) - 2)
        current_node = path[mut_point]
        
        # Get neighbors of previous node
        neighbors = list(self.graph.neighbors(path[mut_point - 1]))
        if not neighbors:
            return None
            
        # Choose a different neighbor
        new_node = random.choice([n for n in neighbors if n != current_node] or neighbors)
        
        # Create new path
        new_path = path[:mut_point] + [new_node] + path[mut_point + 1:]
        
        # Ensure path remains valid
        if self._is_valid_path(new_path):
            return new_path
        return None

    def _segment_mutation(self, path: List[str], origin: str, destination: str) -> Optional[List[str]]:
        """Replace a segment of the path with a random walk"""
        if len(path) <= 3:
            return None
            
        # Select start and end points for segment replacement
        start = random.randint(1, len(path) - 3)
        end = random.randint(start + 1, len(path) - 2)
        
        # Generate random walk for the segment
        segment = self._random_walk(path[start], path[end], max_steps=10)
        if not segment or len(segment) < 2:
            return None
            
        # Replace the segment
        new_path = path[:start] + segment + path[end + 1:]
        
        # Ensure path remains valid
        if self._is_valid_path(new_path):
            return new_path
        return None

    def _scramble_mutation(self, path: List[str], origin: str, destination: str) -> Optional[List[str]]:
        """Scramble a segment of the path"""
        if len(path) <= 3:
            return None
            
        # Select segment to scramble (keep first and last nodes)
        start = random.randint(1, len(path) - 3)
        end = random.randint(start + 1, len(path) - 2)
        
        # Scramble the segment
        segment = path[start:end + 1]
        random.shuffle(segment)
        
        # Ensure segment remains connected
        for i in range(len(segment) - 1):
            if not self.graph.has_edge(segment[i], segment[i + 1]):
                return None
                
        # Rebuild path
        new_path = path[:start] + segment + path[end + 1:]
        return new_path

    def _is_valid_path(self, path: List[str]) -> bool:
        """Check if all edges in path exist"""
        if not path or len(path) < 2:
            return False
            
        for i in range(len(path) - 1):
            if not self.graph.has_edge(path[i], path[i + 1]):
                return False
                
        return True

    def _calculate_path_metrics(self, path: List[str]) -> Dict[str, float]:
        """Calculate cost, time and distance for path"""
        if not path or len(path) < 2:
            return {'cost': float('inf'), 'time': float('inf'), 'distance': float('inf')}
            
        cost = time = distance = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.graph.has_edge(u, v):
                edge = self.graph[u][v]
                cost += edge.get('cost', edge['distance'] * 0.1)
                time += edge['time']
                distance += edge['distance']
        
        return {'cost': cost, 'time': time, 'distance': distance}

    def _get_path_modes(self, path: List[str]) -> List[str]:
        """Get sequence of transportation modes used in path"""
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