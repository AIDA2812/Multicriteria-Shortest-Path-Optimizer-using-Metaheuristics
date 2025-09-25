# solution_graphs.py
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.ticker as ticker
import math

class SolutionGraphs:
    def __init__(self, parent, solutions, algorithm_name):
        self.parent = parent
        self.algorithm_name = algorithm_name
        self.bg_color = '#f8f9fa'
        self.text_bg = '#ffffff'
        
        # Process and validate solutions
        self.solutions = self.process_solutions(solutions)
        
        # Calculate metrics
        self.ideal_point = self.calculate_ideal_point()
        self.euclidean_distances = self.calculate_euclidean_distances()
        self.min_distance_index = np.argmin(self.euclidean_distances) if self.euclidean_distances else -1
        
        self.create_window()
    
    def process_solutions(self, solutions):
        """Ensure all solutions have required fields with proper types and valid values"""
        processed = []
        for sol in solutions:
            try:
                # Handle cost
                cost = sol.get('cost', 0)
                if isinstance(cost, str):
                    if cost.lower() == 'inf':
                        continue  # Skip infinite cost solutions
                    cost = float(cost)
                cost = max(0, float(cost))
                
                # Handle travel_time
                travel_time = sol.get('travel_time', sol.get('time', 0))
                if isinstance(travel_time, str):
                    if travel_time.lower() == 'inf':
                        continue  # Skip infinite time solutions
                    travel_time = float(travel_time)
                travel_time = max(0, float(travel_time))
                
                # Handle distance
                distance = sol.get('distance', 0)
                if isinstance(distance, str):
                    if distance.lower() == 'inf':
                        continue  # Skip infinite distance solutions
                    distance = float(distance)
                distance = max(0, float(distance))
                
                # Check for infinity or NaN values
                if math.isinf(cost) or math.isinf(travel_time) or math.isinf(distance):
                    continue
                if math.isnan(cost) or math.isnan(travel_time) or math.isnan(distance):
                    continue
                
                processed_sol = {
                    'cost': cost,
                    'travel_time': travel_time,
                    'distance': distance,
                    'path': sol.get('path', []),
                    'modes': sol.get('modes', [])
                }
                processed.append(processed_sol)
            except (ValueError, TypeError):
                continue  # Skip invalid solutions
        return processed
    
    def calculate_ideal_point(self):
        """Calculate the ideal point as the minimum of each objective"""
        if not self.solutions:
            return None
        
        return {
            'cost': min(s['cost'] for s in self.solutions),
            'travel_time': min(s['travel_time'] for s in self.solutions),
            'distance': min(s['distance'] for s in self.solutions),
        }
    
    def calculate_euclidean_distances(self):
        """Calculate Euclidean distance between each solution and the ideal point"""
        if not self.solutions or not self.ideal_point:
            return []
            
        distances = []
        for s in self.solutions:
            # Normalize values to avoid scale differences
            cost_norm = (s['cost'] - self.ideal_point['cost']) / max(1, self.ideal_point['cost'])
            time_norm = (s['travel_time'] - self.ideal_point['travel_time']) / max(1, self.ideal_point['travel_time'])
            dist_norm = (s['distance'] - self.ideal_point['distance']) / max(1, self.ideal_point['distance'])
            
            distance = np.sqrt(cost_norm**2 + time_norm**2 + dist_norm**2)
            distances.append(distance)
        
        return distances
    
    def calculate_axis_steps(self, values):
        """Calculate axis steps based on average difference between sorted values"""
        if not values:
            return 1.0
        
        unique_values = sorted(set(values))
        
        # If only one unique value, return a reasonable step
        if len(unique_values) == 1:
            return max(0.1, abs(unique_values[0] * 0.1))
        
        # Calculate differences between consecutive sorted values
        differences = []
        for i in range(1, len(unique_values)):
            diff = unique_values[i] - unique_values[i-1]
            if diff > 1e-9:  # Ignore negligible differences
                differences.append(diff)
        
        # If no meaningful differences found (all values nearly identical)
        if not differences:
            return max(0.1, abs(unique_values[0] * 0.1))
        
        # Calculate average difference
        avg_diff = sum(differences) / len(differences)
        
        # Round to a nice number for step size
        exponent = np.floor(np.log10(avg_diff))
        fraction = avg_diff / (10 ** exponent)
        
        if fraction < 1.5:
            nice_step = 1 * 10**exponent
        elif fraction < 3:
            nice_step = 2 * 10**exponent
        elif fraction < 7:
            nice_step = 5 * 10**exponent
        else:
            nice_step = 10 * 10**exponent
        
        # Ensure minimum step size
        return max(nice_step, 0.1)
    
    def create_window(self):
        """Create the visualization window"""
        self.window = tk.Toplevel(self.parent)
        self.window.title(f"{self.algorithm_name} - Solution Analysis")
        self.window.geometry("1100x750")
        self.window.configure(bg=self.bg_color)
        
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        if not self.solutions:
            self.create_empty_tab()
        else:
            self.create_3d_graph_tab()
            self.create_summary_tab()
    
    def create_3d_graph_tab(self):
        """Create 3D scatter plot showing each unique solution exactly once"""
        tab = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(tab, text="3D View")
        
        fig = plt.figure(figsize=(9, 7), facecolor=self.bg_color)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get unique solutions based on all three metrics
        unique_solutions = []
        seen = set()
        
        for idx, sol in enumerate(self.solutions):
            solution_key = (sol['travel_time'], sol['cost'], sol['distance'])
            if solution_key not in seen:
                seen.add(solution_key)
                unique_solutions.append({
                    'x': sol['travel_time'],
                    'y': sol['cost'],
                    'z': sol['distance'],
                    'indices': [idx],
                    'sol': sol
                })
            else:
                # Add to existing solution's indices
                for us in unique_solutions:
                    if (us['x'], us['y'], us['z']) == solution_key:
                        us['indices'].append(idx)
                        break
        
        # Extract coordinates for plotting
        x = [s['x'] for s in unique_solutions]
        y = [s['y'] for s in unique_solutions]
        z = [s['z'] for s in unique_solutions]
        
        # Calculate axis steps
        x_step = self.calculate_axis_steps(x)
        y_step = self.calculate_axis_steps(y)
        z_step = self.calculate_axis_steps(z)
        
        # Plot each unique solution
        scatter_size = 80
        ax.scatter(x, y, z, c='#1f77b4', alpha=0.8, s=scatter_size, 
                  edgecolors='white', linewidth=0.8)
        
        # Add solution labels showing all indices for each point
        for solution in unique_solutions:
            label = ",".join(str(i+1) for i in solution['indices'])
            ax.text(solution['x'], solution['y'], solution['z'], label,
                   ha='center', va='center', fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.8, pad=1, edgecolor='none'))
        
        # Set axis limits with padding
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)
        z_min, z_max = min(z), max(z)
        
        x_padding = max(x_step, (x_max - x_min) * 0.2)
        y_padding = max(y_step, (y_max - y_min) * 0.2)
        z_padding = max(z_step, (z_max - z_min) * 0.2)
        
        ax.set_xlim(max(0, x_min - x_padding), x_max + x_padding)
        ax.set_ylim(max(0, y_min - y_padding), y_max + y_padding)
        ax.set_zlim(max(0, z_min - z_padding), z_max + z_padding)
        
        # Set axis steps
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_step))
        ax.zaxis.set_major_locator(ticker.MultipleLocator(z_step))
        
        # Highlight optimal solution if it exists
        if self.min_distance_index != -1:
            # Find which unique solution contains the optimal index
            for solution in unique_solutions:
                if self.min_distance_index in solution['indices']:
                    ax.scatter(solution['x'], solution['y'], solution['z'],
                             c='#2ca02c', s=scatter_size*2, marker='o',
                             edgecolor='white', linewidth=1.2, 
                             label='Optimal Solution')
                    ax.text(solution['x'], solution['y'], solution['z'], '★',
                           ha='center', va='center', fontsize=12, color='white')
                    
                    # Draw distance line to ideal point if different
                    if self.ideal_point and (solution['x'] != self.ideal_point['travel_time'] or
                                           solution['y'] != self.ideal_point['cost'] or
                                           solution['z'] != self.ideal_point['distance']):
                        ax.plot([self.ideal_point['travel_time'], solution['x']],
                               [self.ideal_point['cost'], solution['y']],
                               [self.ideal_point['distance'], solution['z']],
                               'r--', alpha=0.8, linewidth=1.5,
                               label=f'Distance: {self.euclidean_distances[self.min_distance_index]:.2f}')
                        
                        # Add distance label
                        mid_x = (self.ideal_point['travel_time'] + solution['x']) / 2
                        mid_y = (self.ideal_point['cost'] + solution['y']) / 2
                        mid_z = (self.ideal_point['distance'] + solution['z']) / 2
                        ax.text(mid_x, mid_y, mid_z,
                               f"Distance: {self.euclidean_distances[self.min_distance_index]:.2f}",
                               color='#d62728', ha='center', va='center', fontsize=9,
                               bbox=dict(facecolor='white', alpha=0.9, pad=2, edgecolor='none'))
                    break
        
        # Plot ideal point
        if self.ideal_point:
            ax.scatter(self.ideal_point['travel_time'], self.ideal_point['cost'], self.ideal_point['distance'],
                      c='#d62728', s=200, marker='*', edgecolor='white', 
                      linewidth=1, label='Ideal Point')
            ax.text(self.ideal_point['travel_time'], self.ideal_point['cost'], self.ideal_point['distance'],
                   '★', ha='center', va='center', fontsize=12, color='white')
        
        # Configure axes
        ax.set_xlabel('Travel Time', fontsize=9, labelpad=8)
        ax.set_ylabel('Cost', fontsize=9, labelpad=8)
        ax.set_zlabel('Distance', fontsize=9, labelpad=8)
        ax.set_title(f"3D Solution Space\n{self.algorithm_name}", pad=20, fontsize=12)
        
        # Add legend
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='upper right', framealpha=0.9)
        
        # Add to GUI
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, tab)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def create_summary_tab(self):
        """Create summary tab showing all solutions and marking identical ones"""
        tab = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(tab, text="Summary")
        
        text = tk.Text(tab, wrap=tk.WORD, font=('Consolas', 10),
                      bg=self.text_bg, fg='#212529', padx=15, pady=15)
        scroll = tk.Scrollbar(tab, command=text.yview)
        text.config(yscrollcommand=scroll.set)
        
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Header
        text.insert(tk.END, f"{self.algorithm_name} Solution Analysis\n", 'header')
        text.insert(tk.END, f"Total Solutions: {len(self.solutions)}\n", 'subheader')
        text.insert(tk.END, "="*60 + "\n\n")
        
        # Configure tags for styling
        text.tag_config('header', font=('Consolas', 12, 'bold'))
        text.tag_config('subheader', font=('Consolas', 10, 'bold'))
        text.tag_config('highlight', foreground='#d62728')
        text.tag_config('identical', foreground='#1f77b4')
        
        # Ideal point
        if self.ideal_point:
            text.insert(tk.END, "Ideal Point (Minima):\n", 'subheader')
            text.insert(tk.END, f"• Cost: ${self.ideal_point['cost']:.2f}\n")
            text.insert(tk.END, f"• Time: {self.ideal_point['travel_time']:.1f} min\n")
            text.insert(tk.END, f"• Distance: {self.ideal_point['distance']:.2f} km\n\n")
        
        # Optimal solution
        if self.min_distance_index != -1:
            opt = self.solutions[self.min_distance_index]
            text.insert(tk.END, f"Optimal Solution (#{self.min_distance_index+1}):\n", 'subheader')
            text.insert(tk.END, f"• Distance to ideal: ", 'highlight')
            text.insert(tk.END, f"{self.euclidean_distances[self.min_distance_index]:.4f}\n")
            text.insert(tk.END, f"• Cost: ${opt['cost']:.2f}\n")
            text.insert(tk.END, f"• Time: {opt['travel_time']:.1f} min\n")
            text.insert(tk.END, f"• Distance: {opt['distance']:.2f} km\n")
            text.insert(tk.END, f"• Path: {' → '.join(opt['path'])}\n")
            text.insert(tk.END, f"• Modes: {' → '.join(opt['modes'])}\n\n")
        
        # Group identical solutions
        solution_groups = []
        used_indices = set()
        
        for i, sol in enumerate(self.solutions):
            if i in used_indices:
                continue
                
            # Find all identical solutions
            identical = [i]
            for j in range(i+1, len(self.solutions)):
                if (sol['cost'] == self.solutions[j]['cost'] and
                    sol['travel_time'] == self.solutions[j]['travel_time'] and
                    sol['distance'] == self.solutions[j]['distance']):
                    identical.append(j)
                    used_indices.add(j)
            
            solution_groups.append({
                'indices': identical,
                'sol': sol
            })
        
        # Display all solutions
        text.insert(tk.END, "All Solutions:\n", 'subheader')
        for group in solution_groups:
            if len(group['indices']) > 1:
                solution_nums = ", ".join(f"#{i+1}" for i in group['indices'])
                text.insert(tk.END, f"\nSolutions {solution_nums} (identical):\n", 'identical')
            else:
                text.insert(tk.END, f"\nSolution {group['indices'][0]+1}:\n", 'highlight')
            
            sol = group['sol']
            text.insert(tk.END, f"• Cost: ${sol['cost']:.2f}\n")
            text.insert(tk.END, f"• Time: {sol['travel_time']:.1f} min\n")
            text.insert(tk.END, f"• Distance: {sol['distance']:.2f} km\n")
            
            # Show individual distances for identical solutions
            if len(group['indices']) > 1:
                text.insert(tk.END, "• Euclidean distances:\n")
                for idx in group['indices']:
                    text.insert(tk.END, f"    - Solution #{idx+1}: {self.euclidean_distances[idx]:.4f}\n")
            else:
                text.insert(tk.END, f"• Euclidean distance: {self.euclidean_distances[group['indices'][0]]:.4f}\n")
            
            text.insert(tk.END, f"• Path: {' → '.join(sol['path'])}\n")
            text.insert(tk.END, f"• Modes: {' → '.join(sol['modes'])}\n")
        
        text.config(state=tk.DISABLED)
    
    def create_empty_tab(self):
        """Create a tab for empty solution case"""
        tab = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(tab, text="No Solutions")
        
        label = tk.Label(tab, text="No valid solutions to display", 
                        font=('Arial', 12), bg=self.bg_color)
        label.pack(pady=50)