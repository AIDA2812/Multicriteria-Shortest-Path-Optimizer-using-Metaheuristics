# generate_data.py
import random

def generate_dataset(num_nodes=100, num_edges=200):
    """Generate dataset with 50 nodes and 600 edges (200 original + 400 duplicates)"""
    # Generate nodes
    nodes = []
    cities = ['New York', 'Chicago', 'Los Angeles', 'San Francisco', 'Atlanta', 'Boston', 'Miami', 'las vigas', 'london', 'Seattle']
    for i in range(1, num_nodes+1):
        city = random.choice(cities)
        node = {
            'id': f"N{i}",
            'label': f"Node {i}",
            'type': random.choice(['subway', 'bus', 'trail']),
            'lat': round(random.uniform(30.0, 45.0), 4),
            'lng': round(random.uniform(-120.0, -75.0), 4),
            'city': city
        }
        nodes.append(node)
    
    # Generate base edges
    node_ids = [node['id'] for node in nodes]
    base_edges = []
    for _ in range(num_edges):
        source = random.choice(node_ids)
        target = random.choice(node_ids)
        while source == target:
            target = random.choice(node_ids)
            
        distance = round(random.uniform(1.0, 50.0), 1)
        time = round(distance * random.uniform(0.8, 1.5), 1)
        cost = round(distance * random.uniform(0.1, 0.5), 2)
        mode = random.choice(['bus', 'subway', 'trail'])
        
        base_edges.append({
            'source': source,
            'target': target,
            'distance': distance,
            'time': time,
            'cost': cost,
            'mode': mode
        })
    
    # Generate non-dominated duplicates (2 per edge)
    all_edges = []
    for edge in base_edges:
        # Original edge
        all_edges.append(edge)
        
        # Duplicate 1: Better cost, worse time/distance
        all_edges.append({
            'source': edge['source'],
            'target': edge['target'],
            'distance': round(edge['distance'] * 1.2, 1),
            'time': round(edge['time'] * 1.3, 1),
            'cost': round(edge['cost'] * 0.7, 2),
            'mode': edge['mode']
        })
        
        # Duplicate 2: Better time, worse cost/distance
        all_edges.append({
            'source': edge['source'],
            'target': edge['target'],
            'distance': round(edge['distance'] * 1.1, 1),
            'time': round(edge['time'] * 0.8, 1),
            'cost': round(edge['cost'] * 1.4, 2),
            'mode': edge['mode']
        })
    
    return nodes, all_edges

def save_dataset(nodes, edges, filename):
    """Save dataset to text file"""
    with open(filename, 'w') as f:
        f.write("# NODES\n")
        f.write("# id,label,type,lat,lng,city\n")
        for node in nodes:
            f.write(f"{node['id']},{node['label']},{node['type']},{node['lat']},{node['lng']},{node['city']}\n")
        
        f.write("\n# EDGES\n")
        f.write("# source,target,distance,time,mode,cost\n")
        for edge in edges:
            f.write(f"{edge['source']},{edge['target']},{edge['distance']},{edge['time']},{edge['mode']},{edge['cost']}\n")

# Generate and save dataset
nodes, edges = generate_dataset(num_nodes=100, num_edges=200)
save_dataset(nodes, edges, "Gnn.txt")