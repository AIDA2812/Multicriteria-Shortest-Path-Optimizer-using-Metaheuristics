import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, StringVar, OptionMenu, Label, LEFT, SOLID
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import networkx as nx
from math import sqrt, ceil
from demo import load_data_from_txt
from optimizer import MultiModalTransportationOptimizer
import webbrowser
from matplotlib.figure import Figure
from solution_graphs import SolutionGraphs
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.id = None
        self.x = self.y = 0
        self.widget.bind("<Enter>", self.showtip)
        self.widget.bind("<Leave>", self.hidetip)
        self.widget.bind("<ButtonPress>", self.hidetip)

    def showtip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        self.tip_window = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        label = Label(tw, text=self.text, justify=LEFT,
                     background="#ffffe0", relief=SOLID, borderwidth=1,
                     font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
        self.tip_window = None

class ModernButton(ttk.Button):
    """Custom styled button with hover effects"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.style = ttk.Style()
        self.style.configure('Modern.TButton', 
                           font=('Arial', 9, 'bold'),
                           borderwidth=1,
                           relief='solid',
                           padding=6,
                           foreground='white',
                           background='#6c757d')
        self.style.map('Modern.TButton',
                      foreground=[('active', 'white')],
                      background=[('active', '#0d6efd')],
                      relief=[('active', 'groove')])
        self.configure(style='Modern.TButton')

class TransportationApp:
    def __init__(self, root):
        # Define color scheme first
        self.bg_color = '#f8f9fa'
        self.sidebar_color = '#e9ecef'
        self.button_color = '#6c757d'
        self.highlight_color = '#0d6efd'
        self.success_color = '#198754'
        self.text_bg = '#ffffff'
        self.disabled_color = '#cccccc'
        
        self.root = root
        self.root.title("Transportation Network Interface")
        self.root.geometry("1200x900")  # Increased window height
        self.root.minsize(1000, 750)   # Increased minimum height
        
        # Modern window styling
        self.root.configure(bg=self.bg_color)
        self.root.option_add('*Font', 'Arial 9')
        
        # Initialize styles
        self.setup_styles()
        
        self.optimizer = MultiModalTransportationOptimizer()
        self.nodes = []
        self.edges = []
        self.graph_loaded = False
        self.current_path = None
        self.zoom_level = 1.0
        self.max_zoom = 3.0
        self.min_zoom = 0.5
        self.fig = None  # Store figure reference for zoom

        # Main container
        self.main_paned = tk.PanedWindow(root, orient=tk.HORIZONTAL, bg=self.bg_color, sashwidth=8, sashrelief=tk.RAISED)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        # Left panel - buttons
        self.left_panel = tk.Frame(self.main_paned, width=150, bg=self.sidebar_color, padx=5, pady=10)
        self.main_paned.add(self.left_panel, minsize=150, stretch="never")

        # Header with logo
        self.header_frame = tk.Frame(self.left_panel, bg=self.sidebar_color)
        self.header_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.logo_label = tk.Label(self.header_frame, text="TransportX", 
                                  font=('Arial', 12, 'bold'), 
                                  bg=self.sidebar_color, fg=self.highlight_color)
        self.logo_label.pack(side=tk.LEFT)
        
        self.version_label = tk.Label(self.header_frame, text="v1.0", 
                                    font=('Arial', 7), 
                                    bg=self.sidebar_color, fg='#6c757d')
        self.version_label.pack(side=tk.RIGHT, padx=5)

        # Buttons frame
        self.button_frame = tk.Frame(self.left_panel, bg=self.sidebar_color)
        self.button_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        buttons = [
    ("📁 Import", self.import_file),
    ("↗ Dijkstra", lambda: self.open_search_window("dijkstra")),
    ("🧬 Genetic", lambda: self.open_search_window("genetic_algorithm")),
    ("🐜 ACO", lambda: self.open_search_window("ant_colony_optimization")),
    ("🐜 ACS", lambda: self.open_search_window("ant_colony_system")),  # Changed label
    ("⚡ OFM", self.optimize_ofm_paths),
    ("🔢 Matrix", self.show_matrix),
    ("🗑️ Clear", self.clear_graph)  
        ]

        for text, cmd in buttons:
            btn = ModernButton(self.button_frame, text=text, command=cmd)
            if text == "⚡ OFM":
                btn.configure(style='Success.TButton')
            btn.pack(side=tk.TOP, pady=4, fill=tk.X)
            ToolTip(btn, self.get_button_tooltip(text.split()[-1]))

        ttk.Separator(self.left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        self.info_btn = ModernButton(self.left_panel, text="ℹ️ About", command=self.show_about)
        self.info_btn.pack(side=tk.BOTTOM, pady=5, fill=tk.X)
        
        # Center panel
        self.center_panel = tk.PanedWindow(self.main_paned, orient=tk.VERTICAL, bg=self.bg_color, sashwidth=8, sashrelief=tk.RAISED)
        self.main_paned.add(self.center_panel, minsize=950)

        # Toolbar frame
        self.toolbar_frame = tk.Frame(self.center_panel, height=40, bg=self.bg_color)
        self.center_panel.add(self.toolbar_frame, minsize=40)

        self.toolbar_buttons_frame = tk.Frame(self.toolbar_frame, bg=self.bg_color)
        self.toolbar_buttons_frame.pack(side=tk.LEFT, padx=5)
        
        self.zoom_in_btn = ModernButton(self.toolbar_buttons_frame, text="+ Zoom In", 
                                      command=self.zoom_in, width=8)
        self.zoom_in_btn.pack(side=tk.LEFT, padx=2)
        
        self.zoom_out_btn = ModernButton(self.toolbar_buttons_frame, text="- Zoom Out", 
                                       command=self.zoom_out, width=8)
        self.zoom_out_btn.pack(side=tk.LEFT, padx=2)
        
        self.status_label = tk.Label(self.toolbar_frame, text="Ready", 
                                    font=('Arial', 8), bg=self.bg_color, fg='#6c757d',
                                    anchor=tk.W)
        self.status_label.pack(side=tk.RIGHT, padx=10, fill=tk.X, expand=True)

        # Graph frame with increased height proportion
        self.graph_frame = tk.Frame(self.center_panel, bg=self.bg_color)
        self.center_panel.add(self.graph_frame, minsize=550)  # Adjusted graph area

        self.legend_frame = tk.Frame(self.graph_frame, bg=self.bg_color)
        self.legend_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        # Text output frame with increased height
        self.text_frame = tk.Frame(self.center_panel, height=300, bg=self.bg_color)  # Increased height
        self.center_panel.add(self.text_frame, minsize=250)  # Increased minimum size
        
        self.notebook = ttk.Notebook(self.text_frame, style='Custom.TNotebook')
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.output_tab = tk.Frame(self.notebook, bg=self.text_bg)
        self.notebook.add(self.output_tab, text="Output")
        
        self.text_area = tk.Text(self.output_tab, wrap=tk.WORD, font=('Consolas', 9),
                               bg=self.text_bg, fg='#212529', insertbackground='#0d6efd',
                               selectbackground='#0d6efd', selectforeground='white',
                               padx=10, pady=10, relief=tk.FLAT)
        self.text_scroll = tk.Scrollbar(self.output_tab, troughcolor=self.bg_color,
                                      bg=self.button_color, activebackground=self.highlight_color)
        self.text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.text_scroll.config(command=self.text_area.yview)
        self.text_area.config(yscrollcommand=self.text_scroll.set)
        
        self.help_tab = tk.Frame(self.notebook, bg=self.text_bg)
        self.notebook.add(self.help_tab, text="Help")
        self.setup_help_tab()

        # Graph variables
        self.canvas = None
        self.toolbar = None
        self.pos = None
        self.city_colors = {}
        self.edge_styles = {
            'bus': {'color': '#1f77b4', 'style': 'dashed', 'width': 1.5},
            'subway': {'color': '#ff7f0e', 'style': 'solid', 'width': 2},
            'trail': {'color': '#2ca02c', 'style': 'dotted', 'width': 1.5}
        }

        # Add interactive elements
        self.setup_button_hover()
        self.setup_window_style()

        self.initialize_empty_graph()
        self.update_buttons_state()
        
        # Bind keyboard shortcuts
        self.root.bind("<Control-o>", lambda e: self.import_file())
        self.root.bind("<Control-l>", lambda e: self.clear_graph())
        self.root.bind("<Control-h>", lambda e: self.show_help())
        self.root.bind("<Control-plus>", lambda e: self.zoom_in())
        self.root.bind("<Control-minus>", lambda e: self.zoom_out())

    def get_button_tooltip(self, text):
        tooltips = {
            "Import": "Load transportation network data from a file (Ctrl+O)",
            "Dijkstra": "Find shortest path using Dijkstra's algorithm",
            "Genetic": "Find path using Genetic Algorithm",
            "ACO": "Find path using Ant Colony Optimization",
            "ACS": "Find path using Ant Colony System",
            "OFM": "Find multiple optimized paths using OFM algorithm",
            "Matrix": "Show transportation matrix between nodes",
            "Clear": "Clear the current graph and all data (Ctrl+L)",
            "About": "Show application information and help"
        }
        return tooltips.get(text, "")

    
    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.style.configure('.', background=self.bg_color)
        self.style.configure('TFrame', background=self.bg_color)
        
        self.style.configure('Modern.TButton', 
                           font=('Arial', 9, 'bold'),
                           borderwidth=1,
                           relief='solid',
                           padding=6,
                           foreground='white',
                           background=self.button_color)
        self.style.map('Modern.TButton',
                      foreground=[('active', 'white')],
                      background=[('active', self.highlight_color)],
                      relief=[('active', 'groove')])
        
        self.style.configure('Success.TButton', 
                           background=self.success_color)
        self.style.map('Success.TButton',
                      background=[('active', '#157347')])
        
        self.style.configure('Custom.TNotebook', background=self.bg_color, borderwidth=0)
        self.style.configure('Custom.TNotebook.Tab', 
                           padding=[10, 4],
                           background=self.sidebar_color,
                           foreground='#495057',
                           font=('Arial', 8, 'bold'))
        self.style.map('Custom.TNotebook.Tab',
                     background=[('selected', self.text_bg), ('active', self.highlight_color)],
                     foreground=[('selected', '#212529'), ('active', 'white')])

    def setup_help_tab(self):
        help_text = """
        Transportation Network Interface Help
        
        Keyboard Shortcuts:
        Ctrl+O - Import file
        Ctrl+L - Clear graph
        Ctrl+H - Show this help
        Ctrl++ - Zoom in
        Ctrl+- - Zoom out
        
        Algorithms:
        1. Dijkstra - Finds the shortest path between nodes
        2. Genetic Algorithm - Evolutionary approach to path finding
        3. Ant Colony Optimization - Inspired by ant behavior
        4. Ant Colony System - Improved ACO algorithm
        5. OFM Optimization - Finds multiple optimized paths
        
        Click the 'About' button for more information.
        """
        
        help_label = tk.Label(self.help_tab, text=help_text, 
                            font=('Consolas', 9), 
                            bg=self.text_bg, fg='#212529',
                            justify=tk.LEFT, anchor=tk.NW)
        help_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def setup_button_hover(self):
        for widget in self.button_frame.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.bind("<Enter>", lambda e: self.status_label.config(
                    text=self.get_button_tooltip(e.widget.cget('text').split()[-1])
                ))
                widget.bind("<Leave>", lambda e: self.status_label.config(text="Ready"))

    def setup_window_style(self):
        self.root.iconbitmap(default='')
        
        menubar = tk.Menu(self.root)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Import", command=self.import_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Clear", command=self.clear_graph, accelerator="Ctrl+L")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Zoom In", command=self.zoom_in, accelerator="Ctrl++")
        view_menu.add_command(label="Zoom Out", command=self.zoom_out, accelerator="Ctrl+-")
        menubar.add_cascade(label="View", menu=view_menu)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Help", command=self.show_help, accelerator="Ctrl+H")
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)

    def zoom_in(self):
        if self.canvas and self.zoom_level < self.max_zoom:
            self.zoom_level *= 1.1
            self.apply_zoom()

    def zoom_out(self):
        if self.canvas and self.zoom_level > self.min_zoom:
            self.zoom_level /= 1.1
            self.apply_zoom()

    def apply_zoom(self):
        if self.canvas and self.fig:
            # Clear the current figure
            self.fig.clf()
            ax = self.fig.add_subplot(111)
            
            # Redraw everything with the new zoom level
            if self.graph_loaded:
                self.redraw_graph(ax)
            else:
                ax.set_title("Transportation Network Grid", 
                            pad=20, fontsize=12, fontweight='bold', color='#212529')
                ax.axis('off')
            
            self.canvas.draw()
    

    def redraw_graph(self, ax):
        """Redraw the graph with current zoom level"""
        graph = self.optimizer.load_data(self.nodes, self.edges)
        
        ax.set_facecolor(self.bg_color)
        ax.set_title("Transportation Network - City Grid Layout", 
                    pad=20, fontsize=12, fontweight='bold', color='#212529')
        ax.axis('off')
        
        # Redraw grid lines if needed
        city_nodes = {}
        for node in self.nodes:
            city = node['city']
            if city not in city_nodes:
                city_nodes[city] = []
            city_nodes[city].append(node)
        
        num_cities = len(city_nodes)
        grid_cols = ceil(sqrt(num_cities))
        city_spacing = 15
        node_spacing = 3
        city_grid_size = 5
        
        # Draw nodes with current colors
        node_colors = [self.city_colors[node['city']] for node in self.nodes]
        nx.draw_networkx_nodes(graph, self.pos, ax=ax, node_size=350,
                             node_color=node_colors, alpha=0.9)
        
        # Draw edges
        for mode, style in self.edge_styles.items():
            edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('mode') == mode]
            nx.draw_networkx_edges(graph, self.pos, edgelist=edges, ax=ax,
                                  edge_color=style['color'],
                                  style=style['style'],
                                  width=style['width'],
                                  alpha=0.7)

        # Draw labels
        nx.draw_networkx_labels(graph, self.pos, ax=ax, 
                              labels={node['id']: node['id'] for node in self.nodes},
                              font_size=8, font_color='black')
        
        # Add city labels
        for i, city in enumerate(city_nodes.keys()):
            row = i // grid_cols
            col = i % grid_cols
            center_x = col * city_spacing + (city_grid_size * node_spacing) / 2
            center_y = row * city_spacing + (city_grid_size * node_spacing) / 2
            ax.text(center_x, center_y, city, fontsize=10, ha='center', va='center', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    def initialize_empty_graph(self):
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('ggplot')
            
        self.fig, ax = plt.subplots(figsize=(10, 7), facecolor=self.bg_color)
        ax.set_facecolor(self.bg_color)
        ax.set_title("Transportation Network Grid", 
                    pad=20, fontsize=12, fontweight='bold', color='#212529')
        ax.axis('off')

        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
            if self.toolbar:
                self.toolbar.destroy()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        self.update_route_legend()

    def update_route_legend(self):
        for widget in self.legend_frame.winfo_children():
            widget.destroy()
            
        legend_title = tk.Label(self.legend_frame, text="Route Types:", 
                              bg=self.bg_color, font=('Arial', 8, 'bold'))
        legend_title.pack(side=tk.LEFT, padx=5)
            
        for mode, style in self.edge_styles.items():
            frame = tk.Frame(self.legend_frame, bg=self.bg_color)
            frame.pack(side=tk.LEFT, padx=5)
            canvas = tk.Canvas(frame, width=40, height=12, bg=self.bg_color, highlightthickness=0)
            canvas.pack(side=tk.LEFT)
            dash_pattern = (4, 2) if style['style'] == 'dashed' else ((1, 2) if style['style'] == 'dotted' else ())
            canvas.create_line(3, 6, 37, 6, fill=style['color'], 
                             width=style['width'], dash=dash_pattern)
            tk.Label(frame, text=mode.capitalize(), font=('Arial', 7), bg=self.bg_color).pack(side=tk.LEFT, padx=2)

    def import_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Transportation File", 
            filetypes=[("Text Files", "*.txt")]
        )
        if file_path:
            try:
                self.show_loading("Loading file...")
                self.nodes, self.edges = load_data_from_txt(file_path)
                self.graph_loaded = True
                messagebox.showinfo(
                    "File Imported", 
                    f"Loaded {len(self.nodes)} nodes and {len(self.edges)} edges."
                )
                self.visualize_graph()
                self.update_buttons_state()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
            finally:
                self.hide_loading()

    def clear_graph(self):
        self.nodes = []
        self.edges = []
        self.graph_loaded = False
        self.current_path = None
        self.initialize_empty_graph()
        self.text_area.delete("1.0", tk.END)
        self.update_buttons_state()
        messagebox.showinfo("Graph Cleared", "The graph has been cleared.")

    def visualize_graph(self):
        if not self.nodes or not self.edges:
            messagebox.showerror("Error", "No data loaded. Please import a file first.")
            return

        graph = self.optimizer.load_data(self.nodes, self.edges)

        # Group nodes by city
        city_nodes = {}
        for node in self.nodes:
            city = node['city']
            if city not in city_nodes:
                city_nodes[city] = []
            city_nodes[city].append(node)

        # Calculate grid dimensions
        num_cities = len(city_nodes)
        grid_cols = ceil(sqrt(num_cities))
        grid_rows = ceil(num_cities / grid_cols)
        
        # Calculate positions
        self.pos = {}
        city_positions = {}
        city_grid_positions = {}
        
        # Spacing parameters
        city_spacing = 15
        node_spacing = 3
        city_grid_size = 5
        
        # Assign positions
        for i, city in enumerate(city_nodes.keys()):
            row = i // grid_cols
            col = i % grid_cols
            city_grid_positions[city] = (col, row)
        
        for city, nodes in city_nodes.items():
            city_col, city_row = city_grid_positions[city]
            
            for j, node in enumerate(nodes):
                sub_col = j % city_grid_size
                sub_row = (j // city_grid_size) % city_grid_size
                
                x = city_col * city_spacing + sub_col * node_spacing
                y = city_row * city_spacing + sub_row * node_spacing
                
                self.pos[node['id']] = (x, y)
                
            center_x = city_col * city_spacing + (city_grid_size * node_spacing) / 2
            center_y = city_row * city_spacing + (city_grid_size * node_spacing) / 2
            city_positions[city] = (center_x, center_y)

        # Create figure with modern styling
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('ggplot')
            
        self.fig, ax = plt.subplots(figsize=(10, 7), facecolor=self.bg_color)
        ax.set_facecolor(self.bg_color)
        
        # Draw grid lines
        self.draw_grid_lines(ax, city_spacing, grid_cols, grid_rows, city_grid_size, node_spacing)
        
        # Create color map
        colors = plt.cm.tab20.colors
        self.city_colors = {city: colors[i % len(colors)] for i, city in enumerate(city_nodes.keys())}
        node_colors = [self.city_colors[node['city']] for node in self.nodes]
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, self.pos, ax=ax, node_size=350,
                             node_color=node_colors, alpha=0.9)
        
        # Draw edges
        for mode, style in self.edge_styles.items():
            edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('mode') == mode]
            nx.draw_networkx_edges(graph, self.pos, edgelist=edges, ax=ax,
                                  edge_color=style['color'],
                                  style=style['style'],
                                  width=style['width'],
                                  alpha=0.7)

        # Draw labels (using node IDs)
        nx.draw_networkx_labels(graph, self.pos, ax=ax, 
                              labels={node['id']: node['id'] for node in self.nodes},
                              font_size=8, font_color='black')
        
        # Add city labels
        for city, (x, y) in city_positions.items():
            ax.text(x, y, city, fontsize=10, ha='center', va='center', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        ax.set_title("Transportation Network - City Grid Layout", 
                    pad=20, fontsize=12, fontweight='bold', color='#212529')
        ax.grid(True, which='both', color='#dee2e6', linestyle='-', linewidth=0.5)
        
        self.update_display()
        self.update_route_legend()

    def draw_grid_lines(self, ax, city_spacing, grid_cols, grid_rows, city_grid_size, node_spacing):
        """Draw grid lines to visualize the structure"""
        # Major grid lines (city boundaries)
        for i in range(grid_cols + 1):
            ax.axvline(i * city_spacing, color='gray', linestyle='-', alpha=0.3)
        for i in range(grid_rows + 1):
            ax.axhline(i * city_spacing, color='gray', linestyle='-', alpha=0.3)
        
        # Minor grid lines (node boundaries)
        for i in range(grid_cols):
            for j in range(city_grid_size + 1):
                x = i * city_spacing + j * node_spacing
                ax.axvline(x, color='lightgray', linestyle=':', alpha=0.2)
        for i in range(grid_rows):
            for j in range(city_grid_size + 1):
                y = i * city_spacing + j * node_spacing
                ax.axhline(y, color='lightgray', linestyle=':', alpha=0.2)

    def update_display(self):
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
            if self.toolbar:
                self.toolbar.destroy()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

    def update_buttons_state(self):
        """Enable/disable buttons based on whether graph is loaded"""
        state = 'normal' if self.graph_loaded else 'disabled'
        
        for widget in self.button_frame.winfo_children():
            if isinstance(widget, ttk.Button) and widget.cget('text') not in ["📁 Import", "🗑️ Clear"]:
                widget.state(['!disabled' if self.graph_loaded else 'disabled'])
                if widget.cget('text') == "⚡ Optimize":
                    widget.configure(style='Success.TButton' if self.graph_loaded else 'Modern.TButton')

    # interface.py (updated open_search_window method)
    # interface.py (updated sections)

    def open_search_window(self, algorithm):
        if not self.graph_loaded:
            messagebox.showerror("Error", "No data loaded. Please import a file first.")
            return

        search_window = Toplevel(self.root)
        search_window.title(f"{algorithm.replace('_', ' ').title()} Search - Select Nodes")
        search_window.configure(bg=self.bg_color)

    # Get all node IDs from imported data
        node_ids = sorted([node['id'] for node in self.nodes])

        start_node_var = StringVar(search_window)
        start_node_var.set(node_ids[0] if node_ids else "")

        end_node_var = StringVar(search_window)
        end_node_var.set(node_ids[1] if len(node_ids) > 1 else "")

        ttk.Label(search_window, text="Select Start Node ID:", background=self.bg_color).pack(pady=5)
        start_menu = ttk.OptionMenu(search_window, start_node_var, node_ids[0], *node_ids)
        start_menu.pack(pady=5)

        ttk.Label(search_window, text="Select End Node ID:", background=self.bg_color).pack(pady=5)
        end_menu = ttk.OptionMenu(search_window, end_node_var, node_ids[1] if len(node_ids) > 1 else node_ids[0], *node_ids)
        end_menu.pack(pady=5)

    # Add criteria selection
        criteria_frame = tk.Frame(search_window, bg=self.bg_color)
        criteria_frame.pack(pady=10)

        ttk.Label(criteria_frame, text="Optimization Criteria:", background=self.bg_color).pack()

    # Create variables for criteria checkboxes
        cost_var = tk.IntVar(value=1 if algorithm == "dijkstra" else 0)
        time_var = tk.IntVar(value=0)
        distance_var = tk.IntVar(value=0)

    # For Dijkstra, only allow one selection
        if algorithm == "dijkstra":
            ttk.Radiobutton(criteria_frame, text="Cost", variable=cost_var, value=1, 
                       command=lambda: [time_var.set(0), distance_var.set(0)]).pack(anchor=tk.W)
            ttk.Radiobutton(criteria_frame, text="Time", variable=time_var, value=1,
                       command=lambda: [cost_var.set(0), distance_var.set(0)]).pack(anchor=tk.W)
            ttk.Radiobutton(criteria_frame, text="Distance", variable=distance_var, value=1,
                       command=lambda: [cost_var.set(0), time_var.set(0)]).pack(anchor=tk.W)
        else:
        # For other algorithms, allow multiple selections with at least one required
            def validate_criteria():
                if not (cost_var.get() or time_var.get() or distance_var.get()):
                    messagebox.showwarning("Warning", "At least one criterion must be selected. Defaulting to cost.")
                    cost_var.set(1)

            ttk.Checkbutton(criteria_frame, text="Cost", variable=cost_var).pack(anchor=tk.W)
            ttk.Checkbutton(criteria_frame, text="Time", variable=time_var).pack(anchor=tk.W)
            ttk.Checkbutton(criteria_frame, text="Distance", variable=distance_var).pack(anchor=tk.W)

    # Add algorithm-specific parameters
        params_frame = tk.Frame(search_window, bg=self.bg_color)
        params_frame.pack(pady=10)

        if algorithm == "genetic_algorithm":
            ga_frame = tk.LabelFrame(params_frame, text="Genetic Algorithm Parameters", bg=self.bg_color)
            ga_frame.pack(pady=5, fill=tk.X)

        # Initialize GA variables as instance attributes
            self.ga_pop_var = tk.IntVar(value=100)
            self.ga_gen_var = tk.IntVar(value=100)
            self.ga_stop_var = tk.IntVar(value=10)
            self.ga_sol_var = tk.IntVar(value=5)

            ttk.Label(ga_frame, text="Population Size:", background=self.bg_color).pack()
            ttk.Entry(ga_frame, textvariable=self.ga_pop_var, width=10).pack()

            ttk.Label(ga_frame, text="Generations:", background=self.bg_color).pack()
            ttk.Entry(ga_frame, textvariable=self.ga_gen_var, width=10).pack()

            ttk.Label(ga_frame, text="Early Stopping:", background=self.bg_color).pack()
            ttk.Entry(ga_frame, textvariable=self.ga_stop_var, width=10).pack()

            ttk.Label(ga_frame, text="Solutions:", background=self.bg_color).pack()
            ttk.Entry(ga_frame, textvariable=self.ga_sol_var, width=10).pack()
        elif algorithm in ["ant_colony_optimization", "ant_colony_system"]:
            aco_frame = tk.LabelFrame(params_frame, text="Ant Colony Parameters", bg=self.bg_color)
            aco_frame.pack(pady=5, fill=tk.X)

        # Common parameters for both ACO and ACS
            ttk.Label(aco_frame, text="Number of Ants:", background=self.bg_color).grid(row=0, column=0, padx=5, pady=2, sticky='w')
            self.aco_ants_var = tk.IntVar(value=10)
            ttk.Entry(aco_frame, textvariable=self.aco_ants_var, width=5).grid(row=0, column=1, padx=5, pady=2, sticky='w')

            ttk.Label(aco_frame, text="Iterations:", background=self.bg_color).grid(row=1, column=0, padx=5, pady=2, sticky='w')
            self.aco_iter_var = tk.IntVar(value=50)
            ttk.Entry(aco_frame, textvariable=self.aco_iter_var, width=5).grid(row=1, column=1, padx=5, pady=2, sticky='w')

            ttk.Label(aco_frame, text="Alpha (pheromone weight):", background=self.bg_color).grid(row=2, column=0, padx=5, pady=2, sticky='w')
            self.aco_alpha_var = tk.DoubleVar(value=1.0)
            ttk.Entry(aco_frame, textvariable=self.aco_alpha_var, width=5).grid(row=2, column=1, padx=5, pady=2, sticky='w')

            ttk.Label(aco_frame, text="Beta (heuristic weight):", background=self.bg_color).grid(row=3, column=0, padx=5, pady=2, sticky='w')
            self.aco_beta_var = tk.DoubleVar(value=2.0)
            ttk.Entry(aco_frame, textvariable=self.aco_beta_var, width=5).grid(row=3, column=1, padx=5, pady=2, sticky='w')

            ttk.Label(aco_frame, text="Evaporation Rate:", background=self.bg_color).grid(row=4, column=0, padx=5, pady=2, sticky='w')
            self.aco_evap_var = tk.DoubleVar(value=0.5)
            ttk.Entry(aco_frame, textvariable=self.aco_evap_var, width=5).grid(row=4, column=1, padx=5, pady=2, sticky='w')

            ttk.Label(aco_frame, text="Number of Solutions:", background=self.bg_color).grid(row=5, column=0, padx=5, pady=2, sticky='w')
            self.aco_sol_var = tk.IntVar(value=3)
            ttk.Entry(aco_frame, textvariable=self.aco_sol_var, width=5).grid(row=5, column=1, padx=5, pady=2, sticky='w')

        # ACS-specific parameters
            if algorithm == "ant_colony_system":
                ttk.Label(aco_frame, text="Exploitation (q0):", background=self.bg_color).grid(row=6, column=0, padx=5, pady=2, sticky='w')
                self.acs_q0_var = tk.DoubleVar(value=0.9)
                ttk.Entry(aco_frame, textvariable=self.acs_q0_var, width=5).grid(row=6, column=1, padx=5, pady=2, sticky='w')

                ttk.Label(aco_frame, text="Local Evaporation (rho):", background=self.bg_color).grid(row=7, column=0, padx=5, pady=2, sticky='w')
                self.acs_rho_var = tk.DoubleVar(value=0.1)
                ttk.Entry(aco_frame, textvariable=self.acs_rho_var, width=5).grid(row=7, column=1, padx=5, pady=2, sticky='w')
 
        def run_search():
            try:
                start_node_id = start_node_var.get()
                end_node_id = end_node_var.get()

            # Verify node IDs exist
                start_node_exists = any(node['id'] == start_node_id for node in self.nodes)
                end_node_exists = any(node['id'] == end_node_id for node in self.nodes)

                if not start_node_exists or not end_node_exists:
                    messagebox.showerror("Error", "Invalid node ID selected.")
                    return

            # Prepare criteria dictionary
                criteria = {
                'cost': bool(cost_var.get()),
                'time': bool(time_var.get()),
                'distance': bool(distance_var.get())
                }

            # If no criteria selected, default to cost
                if not any(criteria.values()):
                    messagebox.showwarning("Warning", "At least one criterion must be selected. Defaulting to cost.")
                    criteria['cost'] = True

            # Run the appropriate algorithm
                if algorithm == "genetic_algorithm":
                    result = self.optimizer.genetic_algorithm(
                    start_node_id, 
                    end_node_id,
                    criteria,
                    population_size=self.ga_pop_var.get(),
                    generations=self.ga_gen_var.get(),
                    early_stopping=self.ga_stop_var.get(),
                    num_solutions=self.ga_sol_var.get()
                    )
                elif algorithm == "ant_colony_optimization":
                    result = self.optimizer.ant_colony_optimization(
                    start_node_id, 
                    end_node_id,
                    criteria=criteria,
                    n_ants=self.aco_ants_var.get(),
                    n_iterations=self.aco_iter_var.get(),
                    alpha=self.aco_alpha_var.get(),
                    beta=self.aco_beta_var.get(),
                    evaporation_rate=self.aco_evap_var.get(),
                    num_solutions=self.aco_sol_var.get()
                    )
                elif algorithm == "ant_colony_system":
                    result = self.optimizer.ant_colony_system(
                    start_node_id, 
                    end_node_id,
                    criteria=criteria,
                    n_ants=self.aco_ants_var.get(),
                    n_iterations=self.aco_iter_var.get(),
                    alpha=self.aco_alpha_var.get(),
                    beta=self.aco_beta_var.get(),
                    evaporation_rate=self.aco_evap_var.get(),
                    q0=self.acs_q0_var.get(),
                    rho=self.acs_rho_var.get(),
                    num_solutions=self.aco_sol_var.get()
                    )
                else:  # dijkstra
                    result = self.optimizer.dijkstra(start_node_id, end_node_id, criteria)

            # Display results
                if algorithm in ["genetic_algorithm", "ant_colony_optimization", "ant_colony_system"]:
                    self.display_multiple_solutions(result, algorithm)
                else:
                    self.display_single_path_result(result, "Dijkstra")

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
            finally:
                search_window.destroy()

        search_button = ttk.Button(search_window, text="Search Path", command=run_search)
        search_button.pack(pady=20)
        
    
    
    def show_solution_space(self, solutions, algorithm_name):
        """Show the solution space visualization window"""
        if not solutions:
            messagebox.showinfo("No Solutions", "No solutions available to visualize.")
            return
        
        
    # Create the graphs window
        SolutionGraphs(self.root, solutions, algorithm_name)
            
    def display_genetic_algorithm_results(self, result):
        """Wrapper for genetic algorithm results to maintain compatibility"""
        self.display_multiple_solutions(result, "Genetic Algorithm")    
    def display_multiple_solutions(self, result, algorithm_name):
        """Display multiple solutions from any algorithm with criteria"""
        self.text_area.delete("1.0", tk.END)

        solutions = result.get('solutions', [])
        if not solutions or not solutions[0].get('path'):
            self.text_area.insert(tk.END, f"No valid paths found using {algorithm_name}.\n")
            return

    # Get optimization criteria - use the criteria from the solution if available
        criteria = solutions[0].get('criteria', {'cost': True})
        active_criteria = [k for k, v in criteria.items() if v]
    
    # Create algorithm-specific header with performance stats
        header = f"{algorithm_name.replace('_', ' ').title()} Results\n"
        header += f"Optimization Criteria: {', '.join(active_criteria) or 'Cost (default)'}\n"
        header += f"Execution Time: {result.get('execution_time', 0):.2f} seconds\n"
        header += f"Solutions Found: {len(solutions)}\n"
        header += "="*50 + "\n\n"
    
    
    # Add algorithm-specific statistics
        if algorithm_name == "genetic_algorithm":
            header += f"Generations: {result.get('num_generations', 'N/A')}\n"
            header += f"Population Size: {result.get('population_size', 'N/A')}\n"
        elif algorithm_name in ["ant_colony_optimization", "ant_colony_system"]:
            header += f"Iterations: {result.get('iterations', 'N/A')}\n"
            if algorithm_name == "ant_colony_system":
                header += f"Exploitation Rate: {result.get('q0', 'N/A')}\n"
                header += f"Diversity Factor: {result.get('diversity_factor', 'N/A')}\n"
    
        self.text_area.insert(tk.END, header)

    # Display each solution with detailed metrics
        for i, solution in enumerate(solutions, 1):
            is_alternative = "+ Mode Adjustment" in solution.get('algorithm', '')
            prefix = "Alternative" if is_alternative else "Solution"
            algorithm_source = solution.get('algorithm', algorithm_name).replace(' + Mode Adjustment', '')
        
            self.text_area.insert(tk.END, f"{prefix} {i} ({algorithm_source}):\n")
            self.text_area.insert(tk.END, f"Path: {' → '.join(solution.get('path', []))}\n")
            self.text_area.insert(tk.END, f"Cost: ${solution.get('cost', 0):.2f}\n")
            self.text_area.insert(tk.END, f"Time: {solution.get('travel_time', solution.get('time', 0)):.1f} min\n")
            self.text_area.insert(tk.END, f"Distance: {solution.get('distance', 0):.2f} km\n")
            self.text_area.insert(tk.END, f"Transport Modes: {' → '.join(solution.get('modes', []))}\n")
        
        # Algorithm-specific metrics
            if algorithm_name == "genetic_algorithm":
                self.text_area.insert(tk.END, f"Fitness: {solution.get('fitness', float('inf')):.4f}\n")
            elif algorithm_name == "ant_colony_system":
                self.text_area.insert(tk.END, f"Pheromone Strength: {solution.get('pheromone', 'N/A')}\n")
        
            self.text_area.insert(tk.END, "\n")

    # Visualize the best path with criteria in title
        if solutions and solutions[0].get('path'):
            best_solution = min(solutions, key=lambda x: x.get('cost', float('inf')))
            title = f"{algorithm_name.replace('_', ' ').title()} Solution"
            if active_criteria:
                title += f" (Optimizing: {', '.join(active_criteria)})"
        
            self.visualize_solution_path(best_solution['path'], title)
        
        # Add performance stats to status bar
            exec_time = result.get('execution_time', result.get('time', 0))
            self.status_label.config(
            text=f"{algorithm_name.replace('_', ' ').title()} completed in {exec_time:.2f}s | " 
                 f"Best cost: ${best_solution.get('cost', 0):.2f}"
            )

    # Add the Visualize Solution Space button
        self.text_area.insert(tk.END, "\n\n")
        self.text_area.window_create(tk.END, window=ttk.Button(
            self.text_area, 
            text="Visualize Solution Space", 
            command=lambda: self.show_solution_space(solutions, algorithm_name.replace('_', ' ').title())
        ))
        self.text_area.insert(tk.END, "\n")

        """Display multiple solutions from any algorithm with criteria"""
        self.text_area.delete("1.0", tk.END)
    
    # Get non-dominated solutions from the result
        solutions = result.get('solutions', [])
        if not solutions:
            self.text_area.insert(tk.END, f"No valid paths found using {algorithm_name}.\n")
            return

    # Get optimization criteria
        criteria = result.get('criteria', {})
        active_criteria = [k for k, v in criteria.items() if v]
    
    # Create algorithm-specific header with performance stats
        header = f"{algorithm_name.replace('_', ' ').title()} Results\n"
        header += f"Optimization Criteria: {', '.join(active_criteria) or 'Cost (default)'}\n"
        header += f"Execution Time: {result.get('execution_time', 0):.2f} seconds\n"
        header += f"Non-Dominated Solutions Found: {len(solutions)}\n"
    
    # Add algorithm-specific statistics
        if algorithm_name == "genetic_algorithm":
            header += f"Generations: {result.get('num_generations', 'N/A')}\n"
            header += f"Population Size: {result.get('population_size', 'N/A')}\n"
        elif algorithm_name in ["ant_colony_optimization", "ant_colony_system"]:
            header += f"Iterations: {result.get('iterations', 'N/A')}\n"
            if algorithm_name == "ant_colony_system":
                header += f"Exploitation Rate: {result.get('q0', 'N/A')}\n"
                header += f"Diversity Factor: {result.get('diversity_factor', 'N/A')}\n"
    
        header += "="*50 + "\n\n"
        self.text_area.insert(tk.END, header)

    # Display each non-dominated solution with detailed metrics
        for i, solution in enumerate(solutions, 1):
            self.text_area.insert(tk.END, f"Solution {i}:\n")
            self.text_area.insert(tk.END, f"Path: {' → '.join(solution.get('path', []))}\n")
            self.text_area.insert(tk.END, f"Cost: ${solution.get('cost', 0):.2f}\n")
            self.text_area.insert(tk.END, f"Time: {solution.get('travel_time', solution.get('time', 0)):.1f} min\n")
            self.text_area.insert(tk.END, f"Distance: {solution.get('distance', 0):.2f} km\n")
            self.text_area.insert(tk.END, f"Transport Modes: {' → '.join(solution.get('modes', []))}\n")
        
        # Algorithm-specific metrics
            if algorithm_name == "genetic_algorithm":
                self.text_area.insert(tk.END, f"Fitness: {solution.get('fitness', float('inf')):.4f}\n")
            elif algorithm_name == "ant_colony_system":
                self.text_area.insert(tk.END, f"Pheromone Strength: {solution.get('pheromone', 'N/A')}\n")
        
            self.text_area.insert(tk.END, "\n")

    # Visualize the first solution with criteria in title
        if solutions and solutions[0].get('path'):
            title = f"{algorithm_name.replace('_', ' ').title()} Solution"
            if active_criteria:
                title += f" (Optimizing: {', '.join(active_criteria)})"
        
            self.visualize_solution_path(solutions[0]['path'], title)
        
        # Add performance stats to status bar
            exec_time = result.get('execution_time', result.get('time', 0))
            self.status_label.config(
            text=f"{algorithm_name.replace('_', ' ').title()} completed in {exec_time:.2f}s | " 
                 f"{len(solutions)} non-dominated solutions"
            )

    # Add the Visualize Solution Space button
        self.text_area.insert(tk.END, "\n\n")
        self.text_area.window_create(tk.END, window=ttk.Button(
        self.text_area, 
        text="Visualize Solution Space", 
        command=lambda: self.show_solution_space(solutions, algorithm_name.replace('_', ' ').title())
        ))
        self.text_area.insert(tk.END, "\n")

    def display_single_path_result(self, result, algorithm_name):
        """Display single path result from other algorithms"""
        self.text_area.delete("1.0", tk.END)

        if not result['path']:
            self.text_area.insert(tk.END, f"No valid path found using {algorithm_name}.\n")
            return

        self.text_area.insert(tk.END, f"{algorithm_name} Results:\n")
        self.text_area.insert(tk.END, f"Execution Time: {result['time']:.4f} seconds\n")  # Added execution time

        self.text_area.insert(tk.END, "="*50 + "\n\n")
        self.text_area.insert(tk.END, f"Path: {' → '.join(result['path'])}\n")
        self.text_area.insert(tk.END, f"Cost: ${result['cost']:.2f}\n")
        self.text_area.insert(tk.END, f"Time: {result['travel_time']:.1f} min\n")
        self.text_area.insert(tk.END, f"Distance: {result['distance']:.2f} km\n")
        self.text_area.insert(tk.END, f"Transport Modes: {' → '.join(result['modes'])}\n")

    # Create a solution format that matches what show_solution_space expects
        solution = {
        'path': result['path'],
        'cost': result['cost'],
        'travel_time': result['travel_time'],
        'distance': result['distance'],
        'modes': result['modes'],
        'criteria': {'cost': True}  # Default criteria for Dijkstra
        }

    # Visualize the path
        self.visualize_solution_path(result['path'], algorithm_name)

    # Add the Visualize Solution Space button
        self.text_area.insert(tk.END, "\n\n")
        self.text_area.window_create(tk.END, window=ttk.Button(
            self.text_area, 
            text="Visualize Solution Space", 
            command=lambda: self.show_solution_space([solution], algorithm_name)
        ))
        self.text_area.insert(tk.END, "\n")
    def visualize_solution_path(self, path, title):
        """Visualize a path solution on the graph with criteria in title"""
        if not path or len(path) < 2:
            return

        graph = self.optimizer.load_data(self.nodes, self.edges)

    # Recreate visualization with path highlight
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('ggplot')
    
        self.fig, ax = plt.subplots(figsize=(10, 7), facecolor=self.bg_color)
        ax.set_facecolor(self.bg_color)

    # Redraw base graph
        node_colors = [self.city_colors[node['city']] for node in self.nodes]

    # Draw all nodes
        nx.draw_networkx_nodes(graph, self.pos, ax=ax, node_size=300, node_color=node_colors, alpha=0.6)

    # Draw all edges (light)
        for mode, style in self.edge_styles.items():
            edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('mode') == mode]
            nx.draw_networkx_edges(graph, self.pos, edgelist=edges, ax=ax,
                         edge_color=style['color'],
                         style=style['style'],
                         width=style['width'],
                         alpha=0.3)

    # Highlight path
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(graph, self.pos, edgelist=path_edges, ax=ax,
                     edge_color='red', width=3, alpha=0.9)
    
        nx.draw_networkx_nodes(graph, self.pos, nodelist=path, ax=ax,
                     node_size=450, node_color='gold',
                     edgecolors='red', linewidths=2)
    
    # Draw labels (using node IDs)
        nx.draw_networkx_labels(graph, self.pos, ax=ax, 
                      labels={node['id']: node['id'] for node in self.nodes},
                      font_size=8, font_color='black')
    
        ax.set_title(title, 
            pad=20, fontsize=12, fontweight='bold', color='#212529')
        ax.grid(True, which='both', color='#dee2e6', linestyle='-', linewidth=0.5)
    
        self.update_display()
    def search_path_by_nodes(self, algorithm, start_node_id, end_node_id, criteria, 
                           pop_size=None, generations=None, ants=None, iterations=None):
        try:
            self.show_loading(f"Finding path using {algorithm.replace('_', ' ')}...")
            self.root.update()  # Force UI update

            if not self.graph_loaded:
                messagebox.showerror("Error", "No data loaded. Please import a file first.")
                return

            # Verify node IDs exist
            start_node_exists = any(node['id'] == start_node_id for node in self.nodes)
            end_node_exists = any(node['id'] == end_node_id for node in self.nodes)
        
            if not start_node_exists or not end_node_exists:
                messagebox.showerror("Error", "Invalid node ID selected.")
                return

            # Call the appropriate algorithm with node IDs and criteria
            if algorithm == "genetic_algorithm":
                result = self.optimizer.genetic_algorithm(
                    start_node_id, 
                    end_node_id, 
                    criteria,
                    population_size=pop_size,
                    generations=generations
                )
            elif algorithm == "ant_colony_optimization":
                result = self.optimizer.ant_colony_optimization(
                    start_node_id, 
                    end_node_id,
                    criteria=criteria,
                    n_ants=ants,
                    n_iterations=iterations
                )
            elif algorithm == "ant_colony_system":
                result = self.optimizer.ant_colony_system(
                    start_node_id, 
                    end_node_id,
                    criteria=criteria,
                    n_ants=ants,
                    n_iterations=iterations
                )
            else:
                result = self.optimizer.dijkstra(start_node_id, end_node_id, criteria)
            
            if not result['path']:
                messagebox.showerror("Error", "No valid path found between the selected nodes.")
                return

            path = result['path']
            modes = result['modes']
            distance = result['distance']
            cost = result['cost']
            travel_time = result.get('travel_time', 'N/A')

            graph = self.optimizer.load_data(self.nodes, self.edges)

            # Recreate visualization with path highlight
            try:
                plt.style.use('seaborn')
            except:
                plt.style.use('ggplot')
                
            self.fig, ax = plt.subplots(figsize=(10, 7), facecolor=self.bg_color)
            ax.set_facecolor(self.bg_color)
            
            # Redraw base graph
            city_nodes = {}
            for node in self.nodes:
                city = node['city']
                if city not in city_nodes:
                    city_nodes[city] = []
                city_nodes[city].append(node)
            
            num_cities = len(city_nodes)
            grid_cols = ceil(sqrt(num_cities))
            
            # Redraw grid lines
            city_spacing = 15
            node_spacing = 3
            city_grid_size = 5
            
            for i in range(grid_cols + 1):
                ax.axvline(i * city_spacing, color='gray', linestyle='-', alpha=0.3)
            for i in range(ceil(num_cities / grid_cols) + 1):
                ax.axhline(i * city_spacing, color='gray', linestyle='-', alpha=0.3)
            
            for i in range(grid_cols):
                for j in range(city_grid_size + 1):
                    x = i * city_spacing + j * node_spacing
                    ax.axvline(x, color='lightgray', linestyle=':', alpha=0.2)
            for i in range(ceil(num_cities / grid_cols)):
                for j in range(city_grid_size + 1):
                    y = i * city_spacing + j * node_spacing
                    ax.axhline(y, color='lightgray', linestyle=':', alpha=0.2)
            
            node_colors = [self.city_colors[node['city']] for node in self.nodes]
            
            # Draw all nodes
            nx.draw_networkx_nodes(graph, self.pos, ax=ax, node_size=300, node_color=node_colors, alpha=0.6)
            
            # Draw all edges (light)
            for mode, style in self.edge_styles.items():
                edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('mode') == mode]
                nx.draw_networkx_edges(graph, self.pos, edgelist=edges, ax=ax,
                                     edge_color=style['color'],
                                     style=style['style'],
                                     width=style['width'],
                                     alpha=0.3)

            # Highlight path
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(graph, self.pos, edgelist=path_edges, ax=ax,
                                 edge_color='red', width=3, alpha=0.9)
            
            nx.draw_networkx_nodes(graph, self.pos, nodelist=path, ax=ax,
                                 node_size=450, node_color='gold',
                                 edgecolors='red', linewidths=2)
            
            # Draw labels (using node IDs)
            nx.draw_networkx_labels(graph, self.pos, ax=ax, 
                                  labels={node['id']: node['id'] for node in self.nodes},
                                  font_size=8, font_color='black')
            
            # Add city labels
            city_positions = {}
            for i, city in enumerate(city_nodes.keys()):
                row = i // grid_cols
                col = i % grid_cols
                center_x = col * city_spacing + (city_grid_size * node_spacing) / 2
                center_y = row * city_spacing + (city_grid_size * node_spacing) / 2
                city_positions[city] = (center_x, center_y)
            
            for city, (x, y) in city_positions.items():
                ax.text(x, y, city, fontsize=10, ha='center', va='center', 
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
            criteria_text = []
            if criteria.get('cost', False):
                criteria_text.append("Cost")
            if criteria.get('time', False):
                criteria_text.append("Time")
            if criteria.get('distance', False):
                criteria_text.append("Distance")
    
            title = f"{algorithm.replace('_', ' ').title()} Path: {start_node_id} → {end_node_id}"
            if criteria_text:
                title += f" (Optimizing: {', '.join(criteria_text)})"

            ax.set_title(title, pad=20, fontsize=12, fontweight='bold', color='#212529')
            ax.grid(True, which='both', color='#dee2e6', linestyle='-', linewidth=0.5)
            
            self.update_display()

            # Update the output text
            details = f"Path Details:\nFrom: {start_node_id}\nTo: {end_node_id}\n "
            details += f"Optimization Criteria: {', '.join(criteria_text) if criteria_text else 'Cost (default)'}\n"
            details += f"Total Distance: {distance:.2f} km\nTotal Cost: ${cost:.2f}\n"
            details += f"Travel Time: {travel_time} minutes\n"
            
            # Add detailed path segment information
            for i in range(len(path)-1):
                u = path[i]
                v = path[i+1]
                edge_data = graph.get_edge_data(u, v)
                details += (f"{u} → {v}: "
                           f"{edge_data['distance']:.2f} km, "
                           f"${edge_data.get('cost', edge_data['distance']*0.1):.2f}, "
                           f"{edge_data['time']} min, "
                           f"mode: {edge_data['mode']}\n")
            
            self.text_area.delete("1.0", tk.END)
            self.text_area.insert(tk.END, details)
            self.text_area.see(tk.END)
            
        finally:
            self.hide_loading()

    def optimize_ofm_paths(self):
        try:
            self.show_loading("Optimizing paths...")

            if not self.graph_loaded:
                messagebox.showerror("Error", "No data loaded. Please import a file first.")
                return

            window = Toplevel(self.root)
            window.title("OFM Algorithm - Select Nodes and Criteria")
            window.configure(bg=self.bg_color)

        # Get all node IDs from imported data
            node_ids = sorted([node['id'] for node in self.nodes])

            start_node_var = StringVar(window)
            start_node_var.set(node_ids[0] if node_ids else "")

            end_node_var = StringVar(window)
            end_node_var.set(node_ids[1] if len(node_ids) > 1 else "")

            ttk.Label(window, text="Select Start Node ID:", background=self.bg_color).pack(pady=5)
            start_menu = ttk.OptionMenu(window, start_node_var, node_ids[0], *node_ids)
            start_menu.pack(pady=5)

            ttk.Label(window, text="Select End Node ID:", background=self.bg_color).pack(pady=5)
            end_menu = ttk.OptionMenu(window, end_node_var, node_ids[1] if len(node_ids) > 1 else node_ids[0], *node_ids)
            end_menu.pack(pady=5)

        # Add criteria selection frame
            criteria_frame = tk.Frame(window, bg=self.bg_color)
            criteria_frame.pack(pady=10)

            ttk.Label(criteria_frame, text="Optimization Criteria:", background=self.bg_color).pack()

            # Create variables for criteria checkboxes
            cost_var = tk.IntVar(value=1)  # Default to cost optimization
            time_var = tk.IntVar(value=0)
            distance_var = tk.IntVar(value=0)
            quality_var = tk.IntVar(value=0)

        # Allow multiple selections for OFM
            ttk.Checkbutton(criteria_frame, text="Cost", variable=cost_var).pack(anchor=tk.W)
            ttk.Checkbutton(criteria_frame, text="Time", variable=time_var).pack(anchor=tk.W)
            ttk.Checkbutton(criteria_frame, text="Distance", variable=distance_var).pack(anchor=tk.W)

        # Add OFM parameters
            params_frame = tk.Frame(window, bg=self.bg_color)
            params_frame.pack(pady=10)

            ttk.Label(params_frame, text="OFM Parameters:", background=self.bg_color).pack()

        # Number of filters
            filters_frame = tk.Frame(params_frame, bg=self.bg_color)
            filters_frame.pack(pady=5)
            ttk.Label(filters_frame, text="Number of filters:", background=self.bg_color).pack(side=tk.LEFT)
            filters_var = tk.IntVar(value=10)
            filters_entry = ttk.Entry(filters_frame, textvariable=filters_var, width=5)
            filters_entry.pack(side=tk.LEFT, padx=5)

        # Number of neighbors
            neighbors_frame = tk.Frame(params_frame, bg=self.bg_color)
            neighbors_frame.pack(pady=5)
            ttk.Label(neighbors_frame, text="Neighbors per filter:", background=self.bg_color).pack(side=tk.LEFT)
            neighbors_var = tk.IntVar(value=5)
            neighbors_entry = ttk.Entry(neighbors_frame, textvariable=neighbors_var, width=5)
            neighbors_entry.pack(side=tk.LEFT, padx=5)

        # Number of iterations
            iter_frame = tk.Frame(params_frame, bg=self.bg_color)
            iter_frame.pack(pady=5)
            ttk.Label(iter_frame, text="Iterations:", background=self.bg_color).pack(side=tk.LEFT)
            iter_var = tk.IntVar(value=20)
            iter_entry = ttk.Entry(iter_frame, textvariable=iter_var, width=5)
            iter_entry.pack(side=tk.LEFT, padx=5)

        # Neighborhood probability
            prob_frame = tk.Frame(params_frame, bg=self.bg_color)
            prob_frame.pack(pady=5)
            ttk.Label(prob_frame, text="Breadth prob (R):", background=self.bg_color).pack(side=tk.LEFT)
            prob_var = tk.DoubleVar(value=0.5)
            prob_entry = ttk.Entry(prob_frame, textvariable=prob_var, width=5)
            prob_entry.pack(side=tk.LEFT, padx=5)

            def run_ofm_algorithm():
                start_node_id = start_node_var.get()
                end_node_id = end_node_var.get()

            # Verify node IDs exist
                start_node_exists = any(node['id'] == start_node_id for node in self.nodes)
                end_node_exists = any(node['id'] == end_node_id for node in self.nodes)

                if not start_node_exists or not end_node_exists:
                    messagebox.showerror("Error", "Invalid node ID selected.")
                    return

            # Prepare criteria dictionary
                criteria = {
                'cost': bool(cost_var.get()),
                'time': bool(time_var.get()),
                'distance': bool(distance_var.get()),
                'quality': bool(quality_var)
                }

            # If no criteria selected, default to cost
                if not any(criteria.values()):
                    criteria['cost'] = True

                result = self.optimizer.ofm_algorithm(
                   start_node_id, 
                end_node_id,
                criteria=criteria,
                num_filters=filters_var.get(),
                num_neighbors=neighbors_var.get(),
                iterations=iter_var.get(),
                R=prob_var.get()
                )
                optimized_paths = result["paths"]
                exec_time = result["time"]  # Get the execution time from the result


                if optimized_paths:
                 # Format the paths for visualization
                    solutions = []
                    for path, dist, time_val, cost, quality in optimized_paths:
                    # Get the modes used in this path
                        modes = []
                        for j in range(len(path)-1):
                            u, v = path[j], path[j+1]
                            if self.optimizer.graph.has_edge(u, v):
                                mode = self.optimizer.graph[u][v]['mode']
                                if not modes or modes[-1] != mode:
                                    modes.append(mode)
                    
                        solutions.append({
                        'path': path,
                        'distance': dist,
                        'travel_time': time_val,
                        'cost': cost,
                        'quality': quality,
                        'modes': modes,
                        'criteria': criteria
                        })

                # Formatw the results for display
                    criteria_text = []
                    if criteria.get('cost', False):
                        criteria_text.append("Cost")
                    if criteria.get('time', False):
                        criteria_text.append("Time")
                    if criteria.get('distance', False):
                        criteria_text.append("Distance")
                    if criteria.get('quality', False):
                        criteria_text.append("Quality")

                    result_text = f"Top Optimized Paths (Optimizing: {', '.join(criteria_text)}):\n\n"
                    result_text += f"Execution Time: {exec_time:.2f} seconds\n\n"  # Add execution time to display

                    for i, solution in enumerate(solutions[:5], 1):
                        result_text += (f"Option {i}:\n"
                                     f"Path: {' → '.join(solution['path'])}\n"
                                  f"Transport Modes: {' → '.join(solution['modes'])}\n"
                                  f"Distance: {solution['distance']:.2f} km\n"
                                  f"Time: {solution['travel_time']:.1f} min\n"
                                  f"Cost: ${solution['cost']:.2f}\n"
                                  f"Quality: {solution['quality']:.2f} (1/transfers)\n\n")

                    # Create a new window to display results
                    result_window = Toplevel(window)
                    result_window.title("OFM Optimization Results")
                    result_window.geometry("800x600")
                    result_window.configure(bg=self.bg_color)

                # Create a frame for the text and button
                    main_frame = tk.Frame(result_window, bg=self.bg_color)
                    main_frame.pack(fill=tk.BOTH, expand=True)

                # Text area with scrollbar
                    text_frame = tk.Frame(main_frame, bg=self.bg_color)
                    text_frame.pack(fill=tk.BOTH, expand=True)

                    text_area = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 9),
                                  bg=self.text_bg, fg='#212529', padx=10, pady=10)
                    scroll = tk.Scrollbar(text_frame, command=text_area.yview,
                                 troughcolor=self.bg_color, bg=self.button_color)
                    text_area.config(yscrollcommand=scroll.set)

                    scroll.pack(side=tk.RIGHT, fill=tk.Y)
                    text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

                    text_area.insert(tk.END, result_text)
                    text_area.config(state=tk.DISABLED)

                # Button frame
                    button_frame = tk.Frame(main_frame, bg=self.bg_color)
                    button_frame.pack(fill=tk.X, pady=10)

                # Add the Visualize Solution Space button
                    ttk.Button(button_frame, 
                          text="Visualize Solution Space", 
                          command=lambda: self.show_solution_space(solutions, "OFM Algorithm")
                          ).pack(pady=5)
                else:
                    messagebox.showerror("Error", "No paths found between the selected nodes.")

            ttk.Button(window, text="Find Optimized Paths", command=run_ofm_algorithm).pack(pady=10)

        finally:
            self.hide_loading()

    def show_matrix(self):
        if not self.graph_loaded:
            messagebox.showerror("Error", "No data loaded. Please import a file first.")
            return

        try:
            self.show_loading("wGenerating transportation matrix...")
            
            # Create a new window for the matrix
            matrix_window = Toplevel(self.root)
            matrix_window.title("Transportation Matrix")
            matrix_window.geometry("800x600")
            
            # Create a frame for the matrix display
            matrix_frame = tk.Frame(matrix_window)
            matrix_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Create a text widget with scrollbars
            text = tk.Text(matrix_frame, wrap=tk.NONE, font=('Consolas', 9))
            vsb = tk.Scrollbar(matrix_frame, orient="vertical", command=text.yview)
            hsb = tk.Scrollbar(matrix_frame, orient="horizontal", command=text.xview)
            text.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            
            # Grid layout
            vsb.pack(side="right", fill="y")
            hsb.pack(side="bottom", fill="x")
            text.pack(side="left", fill="both", expand=True)
            
            # Generate the matrix
            node_ids = sorted([node['id'] for node in self.nodes])
            graph = self.optimizer.load_data(self.nodes, self.edges)
            
            # Header row
            header = "From\\To".ljust(10)
            for node in node_ids:
                header += node.ljust(10)
            text.insert(tk.END, header + "\n")
            
            # Matrix rows
            for from_node in node_ids:
                row = from_node.ljust(10)
                for to_node in node_ids:
                    if from_node == to_node:
                        row += "-".ljust(10)
                    else:
                        edge_data = graph.get_edge_data(from_node, to_node)
                        if edge_data:
                            row += edge_data['mode'][0].upper().ljust(10)
                        else:
                            row += "".ljust(10)
                text.insert(tk.END, row + "\n")
            
            # Add legend
            text.insert(tk.END, "\nLegend:\n")
            for mode in self.edge_styles.keys():
                text.insert(tk.END, f"{mode[0].upper()} = {mode}\n")
            
            text.config(state=tk.DISABLED)
            
        finally:
            self.hide_loading()

    def show_about(self):
        about_window = Toplevel(self.root)
        about_window.title("About Transportation Network Interface")
        about_window.geometry("400x300")
        about_window.resizable(False, False)
        about_window.transient(self.root)
        about_window.grab_set()
        
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 200
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 150
        about_window.geometry(f"+{x}+{y}")
        
        about_frame = tk.Frame(about_window, bg=self.bg_color)
        about_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(about_frame, text="Transportation Network Interface", 
                font=('Arial', 12, 'bold'), bg=self.bg_color).pack(pady=10)
        
        tk.Label(about_frame, text="Version 1.0", 
                font=('Arial', 9), bg=self.bg_color).pack()
        
        tk.Label(about_frame, text="\nThis application provides visualization and analysis\n"
                "tools for multimodal transportation networks.\n", 
                font=('Arial', 9), bg=self.bg_color).pack()
        
        link = tk.Label(about_frame, text="Visit our website", 
                       font=('Arial', 9, 'underline'), 
                       fg='blue', cursor="hand2", bg=self.bg_color)
        link.pack(pady=5)
        link.bind("<Button-1>", lambda e: webbrowser.open("https://example.com"))
        
        ttk.Button(about_frame, text="Close", command=about_window.destroy).pack(pady=10)

    def show_help(self):
        self.notebook.select(self.help_tab)

    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update()

    def show_loading(self, message="Processing..."):
        self.loading_window = Toplevel(self.root)
        self.loading_window.title("Loading")
        self.loading_window.geometry("300x100")
        self.loading_window.resizable(False, False)
        self.loading_window.transient(self.root)
        self.loading_window.grab_set()
        
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 150
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 50
        self.loading_window.geometry(f"+{x}+{y}")
        
        ttk.Label(self.loading_window, text=message, font=('Arial', 10)).pack(pady=10)
        
        self.loading_progress = ttk.Progressbar(
            self.loading_window, 
            mode='indeterminate',
            length=200,
            style='Modern.Horizontal.TProgressbar'
        )
        self.loading_progress.pack(pady=5)
        self.loading_progress.start(10)
        
        self.loading_window.lift()
        self.root.update()

    def hide_loading(self):
        if hasattr(self, 'loading_window') and self.loading_window:
            self.loading_progress.stop()
            self.loading_window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = TransportationApp(root)
    root.mainloop()