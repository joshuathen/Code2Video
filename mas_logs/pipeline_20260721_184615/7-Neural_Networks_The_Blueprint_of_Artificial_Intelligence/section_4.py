from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section4Scene(TeachingScene):
    def construct(self):
        # Colors (Hexadecimal as per L008)
        COLOR_INPUT = "#5555FF"
        COLOR_HIDDEN = "#FFFFFF"
        COLOR_OUTPUT = "#00FF00"
        COLOR_CYAN = "#00FFFF"
        COLOR_YELLOW = "#FFCC00"
        COLOR_INACTIVE = "#666666"

        title = "The Architecture: Layers and Depth"
        # Lecture lines
        lines = [
            "- Neurons organize into input, hidden, and output layers.",
            "- Multiple hidden layers create what we call deep learning.",
            "- Each layer extracts increasingly complex features from the data."
        ]
        self.setup_layout(title, lines)
        
        # Initial state for lecture lines
        self.lecture.set_color(COLOR_INACTIVE)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE), run_time=0.5)
        
        # Input Layer - Adjusted for better spatial utilization (Issue 33, 42)
        input_nodes = VGroup(*[Circle(radius=0.15, color=COLOR_INPUT, fill_opacity=1, stroke_width=0) for _ in range(3)])
        self.place_at_grid(input_nodes[0], "B1")
        self.place_at_grid(input_nodes[1], "C1")
        self.place_at_grid(input_nodes[2], "D1")
        
        input_label = Text("Input", font_size=20, color=COLOR_INPUT)
        self.place_at_grid(input_label, "A1", scale_factor=0.8) # Adjusted (Issue 33, 42)
        
        # Hidden Layer 1 - Adjusted column to match label (Issue 32, 42)
        hidden_nodes_1 = VGroup(*[Circle(radius=0.15, color=COLOR_HIDDEN, fill_opacity=1, stroke_width=0) for _ in range(4)])
        self.place_at_grid(hidden_nodes_1[0], "B2")
        self.place_at_grid(hidden_nodes_1[1], "C2")
        self.place_at_grid(hidden_nodes_1[2], "D2")
        self.place_at_grid(hidden_nodes_1[3], "E2")
        
        hidden_label = Text("Hidden", font_size=20, color=COLOR_HIDDEN)
        self.place_at_grid(hidden_label, "A2", scale_factor=0.8) # Adjusted (Issue 32, 42)
        
        # Output Layer - Adjusted for proximity (Issue 31, 42)
        output_nodes = VGroup(*[Circle(radius=0.15, color=COLOR_OUTPUT, fill_opacity=1, stroke_width=0) for _ in range(1)])
        self.place_at_grid(output_nodes[0], "B6") # Closer to A6
        
        output_label = Text("Output", font_size=20, color=COLOR_OUTPUT)
        self.place_at_grid(output_label, "A6", scale_factor=0.8)
        
        # Connections helper
        def get_connections(layer1, layer2):
            conns = VGroup()
            for n1 in layer1:
                for n2 in layer2:
                    conns.add(Line(n1.get_center(), n2.get_center(), stroke_width=1, color="#666666", stroke_opacity=0.3))
            return conns
            
        conns_1 = get_connections(input_nodes, hidden_nodes_1)
        conns_2 = get_connections(hidden_nodes_1, output_nodes)
        
        self.play(FadeIn(input_nodes), FadeIn(input_label), run_time=1)
        self.play(FadeIn(hidden_nodes_1), FadeIn(hidden_label), run_time=1)
        self.play(FadeIn(output_nodes), FadeIn(output_label), run_time=1)
        self.play(Create(conns_1), Create(conns_2), run_time=1.5)
        
        self.wait(2.0)
        
        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(COLOR_INACTIVE),
            self.lecture[1].animate.set_color(WHITE),
            run_time=0.5
        )
        
        # Prep new layers for Deep Learning visualization (Shifted to Col 3, 4, 5)
        h1_new = VGroup(*[Circle(radius=0.15, color=COLOR_HIDDEN, fill_opacity=1, stroke_width=0) for _ in range(4)])
        self.place_at_grid(h1_new[0], "B3")
        self.place_at_grid(h1_new[1], "C3")
        self.place_at_grid(h1_new[2], "D3")
        self.place_at_grid(h1_new[3], "E3")

        h2_new = VGroup(*[Circle(radius=0.15, color=COLOR_HIDDEN, fill_opacity=1, stroke_width=0) for _ in range(4)])
        self.place_at_grid(h2_new[0], "B4")
        self.place_at_grid(h2_new[1], "C4")
        self.place_at_grid(h2_new[2], "D4")
        self.place_at_grid(h2_new[3], "E4")

        h3_new = VGroup(*[Circle(radius=0.15, color=COLOR_HIDDEN, fill_opacity=1, stroke_width=0) for _ in range(4)])
        self.place_at_grid(h3_new[0], "B5")
        self.place_at_grid(h3_new[1], "C5")
        self.place_at_grid(h3_new[2], "D5")
        self.place_at_grid(h3_new[3], "E5")

        # Define connections for expanded network
        conns_d1 = get_connections(input_nodes, h1_new)
        conns_d2 = get_connections(h1_new, h2_new)
        conns_d3 = get_connections(h2_new, h3_new)
        conns_d4 = get_connections(h3_new, output_nodes)

        deep_label = Text("Deep Learning", font_size=20, color=COLOR_HIDDEN)
        # Positioned spanning the area above the 3 hidden layers (Issue 32, 42)
        self.place_in_area(deep_label, "A3", "A5", scale_factor=0.8)

        # Transition to Deep Architecture
        self.play(
            FadeOut(conns_1), FadeOut(conns_2), FadeOut(hidden_label),
            ReplacementTransform(hidden_nodes_1, h1_new),
            FadeIn(h2_new), FadeIn(h3_new),
            FadeIn(deep_label),
            run_time=1.5
        )
        self.play(Create(conns_d1), Create(conns_d2), Create(conns_d3), Create(conns_d4), run_time=1.5)
        self.wait(2.0)
        
        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(COLOR_INACTIVE),
            self.lecture[2].animate.set_color(COLOR_CYAN),
            run_time=0.5
        )
        
        # Flash cyan on the first layer (Input) to show feature extraction start
        # Use Indicate as per L004
        self.play(Indicate(input_nodes, color=COLOR_CYAN, scale_factor=1.2), run_time=1.5)
        self.wait(1.5) 
        
        # Flash yellow on the first hidden layer to show hierarchical progression
        self.play(Indicate(h1_new, color=COLOR_YELLOW, scale_factor=1.2), run_time=1.5)
        
        self.wait(1.5)
