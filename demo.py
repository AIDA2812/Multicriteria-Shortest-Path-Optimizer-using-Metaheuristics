import json
import matplotlib.pyplot as plt
import os
import numpy as np
from optimizer import MultiModalTransportationOptimizer

def load_sample_data():
    """Load sample transportation data with city information"""
    # Sample data with city information
    nodes = [
        {"id": "A", "label": "Central Station", "type": "subway", "lat": 40.7128, "lng": -74.0060, "city": "New York"},
        {"id": "B", "label": "Bus Terminal", "type": "bus", "lat": 40.7282, "lng": -73.9942, "city": "New York"},
        {"id": "C", "label": "North Station", "type": "subway", "lat": 40.7589, "lng": -73.9851, "city": "New York"},
        {"id": "D", "label": "Ferry Terminal", "type": "bus", "lat": 40.7061, "lng": -74.0119, "city": "New York"},
        {"id": "E", "label": "Downtown Station", "type": "bus", "lat": 41.8781, "lng": -87.6298, "city": "Chicago"},
        {"id": "F", "label": "Airport Connection", "type": "bus", "lat": 41.8339, "lng": -87.6722, "city": "Chicago"},
        {"id": "G", "label": "Lakefront Trail", "type": "trail", "lat": 41.8916, "lng": -87.6079, "city": "Chicago"},
        {"id": "H", "label": "Union Station", "type": "subway", "lat": 41.8786, "lng": -87.6403, "city": "Chicago"},
        {"id": "I", "label": "Central Hub", "type": "subway", "lat": 34.0522, "lng": -118.2437, "city": "Los Angeles"},
        {"id": "J", "label": "Beach Station", "type": "bus", "lat": 34.0194, "lng": -118.4912, "city": "Los Angeles"},
        {"id": "K", "label": "Downtown Terminal", "type": "bus", "lat": 34.0407, "lng": -118.2468, "city": "Los Angeles"},
        {"id": "L", "label": "Metro Center", "type": "subway", "lat": 34.0487, "lng": -118.2590, "city": "Los Angeles"},
    ]
    
    edges = [
        {"source": "A", "target": "B", "distance": 2.5, "time": 10, "mode": "bus", "cost": 2.75},
        {"source": "A", "target": "C", "distance": 4.2, "time": 15, "mode": "subway", "cost": 3.50},
        {"source": "B", "target": "C", "distance": 3.1, "time": 12, "mode": "bus", "cost": 2.75},
        {"source": "B", "target": "D", "distance": 2.8, "time": 11, "mode": "bus", "cost": 2.75},
        {"source": "C", "target": "D", "distance": 1.9, "time": 8, "mode": "bus", "cost": 2.75},
        {"source": "E", "target": "F", "distance": 3.7, "time": 20, "mode": "bus", "cost": 2.50},
        {"source": "E", "target": "G", "distance": 2.2, "time": 9, "mode": "bus", "cost": 2.50},
        {"source": "F", "target": "H", "distance": 1.5, "time": 6, "mode": "bus", "cost": 2.50},
        {"source": "G", "target": "H", "distance": 4.0, "time": 35, "mode": "trail", "cost": 0.00},
        {"source": "I", "target": "J", "distance": 8.5, "time": 30, "mode": "bus", "cost": 1.75},
        {"source": "I", "target": "K", "distance": 2.1, "time": 12, "mode": "subway", "cost": 1.75},
        {"source": "J", "target": "L", "distance": 5.3, "time": 22, "mode": "bus", "cost": 1.75},
        {"source": "K", "target": "L", "distance": 1.1, "time": 4, "mode": "bus", "cost": 1.75},
        # Inter-city connections
        {"source": "C", "target": "E", "distance": 1200, "time": 150, "mode": "subway", "cost": 120.00},
        {"source": "H", "target": "I", "distance": 2800, "time": 240, "mode": "subway", "cost": 280.00},
    ]
    
    return nodes, edges

def load_data_from_json(file_path):
    """Load transportation data from a JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data.get('nodes', []), data.get('edges', [])

def load_data_from_txt(file_path):
    """Load transportation data from a text file
    
    Expected format:
    # NODES
    # id,label,type,lat,lng,city
    A,Central Station,subway,40.7128,-74.0060,New York
    B,Bus Terminal,bus,40.7282,-73.9942,New York
    ...
    # EDGES
    # source,target,distance,time,mode,cost
    A,B,2.5,10,bus,2.75
    A,C,4.2,15,subway,3.50
    ...
    """
    nodes = []
    edges = []
    current_section = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments that don't indicate sections
            if not line or (line.startswith('#') and not line.lower() in ['# nodes', '# edges']):
                continue
            
            # Determine which section we're in
            if line.lower() == '# nodes':
                current_section = 'nodes'
                continue
            elif line.lower() == '# edges':
                current_section = 'edges'
                continue
            
            # Process data based on current section
            if current_section == 'nodes':
                if not line.startswith('#'):  # Skip header line
                    parts = line.split(',')
                    if len(parts) >= 6:
                        node = {
                            "id": parts[0],
                            "label": parts[1],
                            "type": parts[2],
                            "lat": float(parts[3]),
                            "lng": float(parts[4]),
                            "city": parts[5]
                        }
                        nodes.append(node)
            
            elif current_section == 'edges':
                if not line.startswith('#'):  # Skip header line
                    parts = line.split(',')
                    if len(parts) >= 6:
                        edge = {
                            "source": parts[0],
                            "target": parts[1],
                            "distance": float(parts[2]),
                            "time": float(parts[3]),
                            "mode": parts[4],
                            "cost": float(parts[5])
                        }
                        edges.append(edge)
    
    return nodes, edges


def create_sample_txt_file(file_path):
    """Create a sample text file with transportation data"""
    nodes, edges = load_sample_data()
    
    with open(file_path, 'w') as f:
        # Write nodes section
        f.write("# NODES\n")
        f.write("# id,label,type,lat,lng,city\n")
        for node in nodes:
            f.write(f"{node['id']},{node['label']},{node['type']},{node['lat']},{node['lng']},{node['city']}\n")
        
        # Write edges section
        f.write("\n# EDGES\n")
        f.write("# source,target,distance,time,mode,cost\n")
        for edge in edges:
            cost = edge.get('cost', edge['distance'] * 0.1)  # Default cost if not provided
            f.write(f"{edge['source']},{edge['target']},{edge['distance']},{edge['time']},{edge['mode']},{cost:.2f}\n")

def compare_algorithms_chart(results, origin_city, destination_city):
    """Create charts comparing algorithm performance"""
    algorithms = list(results.keys())
    costs = [results[alg]['cost'] for alg in algorithms]
    computation_times = [results[alg]['time'] * 1000 for alg in algorithms]  # Convert to milliseconds
    travel_times = [results[alg]['travel_time'] for alg in algorithms]
    distances = [results[alg]['distance'] for alg in algorithms]
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Algorithm Performance Comparison: {origin_city} to {destination_city}', fontsize=16)
    
    # 1. Cost comparison
    axs[0, 0].bar(algorithms, costs, color='blue', alpha=0.7)
    axs[0, 0].set_ylabel('Cost ($)', color='blue')
    axs[0, 0].set_title('Route Cost Comparison')
    
    # 2. Computation time comparison
    axs[0, 1].bar(algorithms, computation_times, color='red', alpha=0.7)
    axs[0, 1].set_ylabel('Computation Time (ms)', color='red')
    axs[0, 1].set_title('Algorithm Execution Time')
    
    # 3. Travel time comparison
    axs[1, 0].bar(algorithms, travel_times, color='green', alpha=0.7)
    axs[1, 0].set_ylabel('Travel Time (min)', color='green')
    axs[1, 0].set_title('Route Travel Time')
    
    # 4. Distance comparison
    axs[1, 1].bar(algorithms, distances, color='purple', alpha=0.7)
    axs[1, 1].set_ylabel('Distance (km)', color='purple')
    axs[1, 1].set_title('Route Distance')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{origin_city}_to_{destination_city}_comparison.png', dpi=300)
    plt.show()

def select_cities(optimizer):
    """Allow user to select origin and destination cities"""
    cities = optimizer.get_cities()
    
    print("\nAvailable cities:")
    for i, city in enumerate(cities):
        print(f"{i+1}. {city}")
    
    # Get origin city
    while True:
        try:
            origin_idx = int(input("\nSelect origin city (number): ")) - 1
            if 0 <= origin_idx < len(cities):
                origin_city = cities[origin_idx]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")
    
    # Get destination city
    while True:
        try:
            dest_idx = int(input("Select destination city (number): ")) - 1
            if 0 <= dest_idx < len(cities) and dest_idx != origin_idx:
                destination_city = cities[dest_idx]
                break
            elif dest_idx == origin_idx:
                print("Destination cannot be the same as origin. Please try again.")
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")
    
    return origin_city, destination_city

def display_detailed_statistics(results, origin_city, destination_city):
    """Display detailed statistics about the calculated routes"""
    print("\n" + "=" * 80)
    print(f"DETAILED ROUTE STATISTICS: {origin_city} TO {destination_city}")
    print("=" * 80)
    
    for algorithm, result in results.items():
        print(f"\n{algorithm.upper()} ALGORITHM RESULTS:")
        print("-" * 50)
        
        if not result['path']:
            print("No valid path found.")
            continue
        
        # Basic statistics
        print(f"Path: {' → '.join(result['path'])}")
        print(f"Total Cost: ${result['cost']:.2f}")
        print(f"Total Travel Time: {result['travel_time']:.1f} minutes")
        print(f"Total Distance: {result['distance']:.2f} km")
        print(f"Algorithm Execution Time: {result['time']*1000:.2f} ms")
        
        # Mode analysis
        modes = result['modes']
        mode_counts = {}
        mode_distances = {}
        mode_times = {}
        mode_costs = {}
        
        # Calculate statistics for each transportation mode
        for i, mode in enumerate(modes):
            source = result['path'][i]
            target = result['path'][i+1]
            
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            
            edge_data = None
            for u, v, data in result['graph'].edges(data=True):
                if u == source and v == target:
                    edge_data = data
                    break
            
            if edge_data:
                mode_distances[mode] = mode_distances.get(mode, 0) + edge_data['distance']
                mode_times[mode] = mode_times.get(mode, 0) + edge_data['time']
                mode_costs[mode] = mode_costs.get(mode, 0) + edge_data.get('cost', edge_data['distance'] * 0.1)
        
        # Print mode statistics
        print("\nTransportation Mode Analysis:")
        print(f"{'Mode':<10} {'Segments':<10} {'Distance (km)':<15} {'Time (min)':<15} {'Cost ($)':<10}")
        print("-" * 60)
        
        for mode in sorted(mode_counts.keys()):
            print(f"{mode:<10} {mode_counts[mode]:<10} {mode_distances[mode]:<15.2f} {mode_times[mode]:<15.1f} {mode_costs[mode]:<10.2f}")
        
        print("-" * 60)
        print(f"TOTAL      {len(modes):<10} {result['distance']:<15.2f} {result['travel_time']:<15.1f} {result['cost']:<10.2f}")
    
    print("\n" + "=" * 80)

def main():
    print("Multi-Modal Transportation Optimizer Demo")
    print("----------------------------------------")
    print("This application finds optimal routes between cities using three algorithms:")
    print("1. Genetic Algorithm")
    print("2. Ant Colony Optimization")
    print("3. Ant Colony System")
    
    # Create optimizer
    optimizer = MultiModalTransportationOptimizer()
    
    # Check if data file exists, create it if it doesn't
    data_file = "transportation_network.txt"
    if not os.path.exists(data_file):
        print(f"\nCreating sample transportation data file: {data_file}")
        create_sample_txt_file(data_file)
    
    # Load data from text file
    print(f"\nLoading transportation data from: {data_file}")
    nodes, edges = load_data_from_txt(data_file)
    
    # Load data into optimizer
    graph = optimizer.load_data(nodes, edges)
    print(f"Loaded graph with {len(nodes)} nodes and {len(edges)} edges")
    
    # Get available cities
    cities = optimizer.get_cities()
    print(f"Available cities: {', '.join(cities)}")
    
    # Allow user to select cities or use defaults
    use_defaults = input("\nUse default cities (New York to Los Angeles)? (y/n): ").lower() == 'y'
    
    if use_defaults:
        origin_city = "New York"
        destination_city = "Los Angeles"
    else:
        origin_city, destination_city = select_cities(optimizer)
    
    print(f"\nSelected route: {origin_city} to {destination_city}")
    
    # Visualize network with highlighted cities
    print("\nVisualizing transportation network...")
    optimizer.visualize_network(
        highlight_cities=[origin_city, destination_city],
        save_path=f"{origin_city}_to_{destination_city}_network.png"
    )
    
    # Compare algorithms for city-to-city routing
    print(f"\nFinding optimal routes from {origin_city} to {destination_city}...")
    print("This may take a moment as the algorithms explore different route possibilities...")
    
    results = optimizer.compare_algorithms(origin_city, destination_city)
    
    # Add graph reference to results for statistics calculation
    for algorithm in results:
        results[algorithm]['graph'] = optimizer.graph
    
    # Print detailed statistics
    display_detailed_statistics(results, origin_city, destination_city)
    
    # Print summary results
    print("\nSummary Results:")
    for algorithm, result in results.items():
        path_summary = "No valid path found."
        if result['path']:
            if len(result['path']) > 6:
                path_segments = [result['path'][0], result['path'][1], "...", result['path'][-2], result['path'][-1]]
                path_summary = f"{' → '.join(path_segments)} ({len(result['path'])} nodes)"
            else:
                path_summary = f"{' → '.join(result['path'])}"
        
        print(f"{algorithm.capitalize()}: Cost = ${result['cost']:.2f}, Travel Time = {result['travel_time']:.1f} min, "
              f"Distance = {result['distance']:.2f} km, Execution Time = {result['time']*1000:.2f} ms")
        print(f"  Path: {path_summary}")
    
    # Create comparison charts
    print("\nGenerating algorithm comparison charts...")
    compare_algorithms_chart(results, origin_city, destination_city)
    
    # Visualize optimal paths
    print("\nVisualizing optimal paths...")
    for algorithm, result in results.items():
        if result['path']:
            print(f"Visualizing {algorithm} path...")
            optimizer.visualize_optimal_path(
                result['path'], 
                highlight_cities=[origin_city, destination_city],
                save_path=f"{origin_city}_to_{destination_city}_{algorithm}_path.png"
            )
    
    print("\nDemo completed. Check the generated visualization files.")

if __name__ == "__main__":
    main()
