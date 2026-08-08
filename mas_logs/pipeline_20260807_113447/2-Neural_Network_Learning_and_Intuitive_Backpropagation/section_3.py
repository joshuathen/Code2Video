from manim import *
import numpy as np

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

class Section3Scene(TeachingScene):
    def construct(self):
        # Colors
        HIGHLIGHT_COLOR = "#FFFF00"
        SIGNAL_COLOR = "#FFFF00"
        NORMAL_NEURON_COLOR = WHITE
        CONNECTION_COLOR = GRAY
        
        lecture_lines = [
            "Data flows from input to output layers.",
            "Multiply inputs by weights and sum them up.",
            "Signals light up neurons across the network.",
            "The final output is a confidence score.",
            "Byte the robot predicts if it's a lemon."
        ]
        
        self.setup_layout("The Forward Pass: Making a Guess", lecture_lines)
        
        # Define Layers and Nodes
        # Adjusted positions to respect B021 and issues 29, 30
        input_pos = ['B2', 'C2', 'D2']
        hidden_pos = ['B3', 'C3', 'D3', 'E3']
        output_pos = ['C5', 'D5']
        
        input_nodes = VGroup(*[Circle(radius=0.15, color=NORMAL_NEURON_COLOR, stroke_width=2) for _ in input_pos])
        hidden_nodes = VGroup(*[Circle(radius=0.15, color=NORMAL_NEURON_COLOR, stroke_width=2) for _ in hidden_pos])
        output_nodes = VGroup(*[Circle(radius=0.15, color=NORMAL_NEURON_COLOR, stroke_width=2) for _ in output_pos])
        
        # Initial placement
        for i, pos in enumerate(input_pos):
            self.place_at_grid(input_nodes[i], pos)
        for i, pos in enumerate(hidden_pos):
            self.place_at_grid(hidden_nodes[i], pos)
        for i, pos in enumerate(output_pos):
            self.place_at_grid(output_nodes[i], pos)
            
        # Create Connections
        in_hidden_connections = VGroup()
        for in_node in input_nodes:
            for h_node in hidden_nodes:
                line = Line(in_node.get_center(), h_node.get_center(), stroke_width=1, color=CONNECTION_COLOR, stroke_opacity=0.3)
                in_hidden_connections.add(line)
                
        hidden_out_connections = VGroup()
        for h_node in hidden_nodes:
            for out_node in output_nodes:
                line = Line(h_node.get_center(), out_node.get_center(), stroke_width=1, color=CONNECTION_COLOR, stroke_opacity=0.3)
                hidden_out_connections.add(line)
        
        self.add(in_hidden_connections, hidden_out_connections, input_nodes, hidden_nodes, output_nodes)
        
        # === Animation for Lecture Line 1 ===
        # "Data flows from input to output layers."
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        self.play(
            *[node.animate.set_fill(HIGHLIGHT_COLOR, opacity=0.8).set_color(HIGHLIGHT_COLOR) for node in input_nodes],
            run_time=1
        )
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # "Multiply inputs by weights and sum them up."
        self.play(self.lecture[1].animate.set_color(HIGHLIGHT_COLOR))
        
        # Create signals for first transition
        signals_1 = VGroup()
        for line in in_hidden_connections:
            dot = Dot(radius=0.03, color=SIGNAL_COLOR).move_to(line.get_start())
            signals_1.add(dot)
            
        self.play(
            *[MoveAlongPath(dot, line) for dot, line in zip(signals_1, in_hidden_connections)],
            run_time=1.5,
            rate_func=linear
        )
        self.remove(signals_1)
        
        # Hidden layer lights up
        self.play(
            *[node.animate.set_fill(HIGHLIGHT_COLOR, opacity=0.5).set_color(HIGHLIGHT_COLOR) for node in hidden_nodes],
            run_time=0.5
        )
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # "Signals light up neurons across the network."
        self.play(self.lecture[2].animate.set_color(HIGHLIGHT_COLOR))
        
        # Create signals for second transition
        signals_2 = VGroup()
        for line in hidden_out_connections:
            dot = Dot(radius=0.03, color=SIGNAL_COLOR).move_to(line.get_start())
            signals_2.add(dot)
            
        self.play(
            *[MoveAlongPath(dot, line) for dot, line in zip(signals_2, hidden_out_connections)],
            run_time=1.5,
            rate_func=linear
        )
        self.remove(signals_2)
        
        # Output layer lights up (initial)
        self.play(
            *[node.animate.set_fill(HIGHLIGHT_COLOR, opacity=0.5).set_color(HIGHLIGHT_COLOR) for node in output_nodes],
            run_time=0.5
        )
        self.wait(1)
        
        # === Animation for Lecture Line 4 ===
        # "The final output is a confidence score."
        self.play(self.lecture[3].animate.set_color(HIGHLIGHT_COLOR))
        
        # Highlight 'Lemon' node specifically (top output node at C5)
        lemon_node = output_nodes[0]
        self.play(
            lemon_node.animate.set_fill(HIGHLIGHT_COLOR, opacity=1.0).scale(1.3),
            run_time=1
        )
        
        # Score label (Issue 29: move from A6 to B6)
        score_text = Text("0.8", font_size=24, color=HIGHLIGHT_COLOR)
        self.place_at_grid(score_text, 'B6')
        self.play(Write(score_text))
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        # "Byte the robot predicts if it's a lemon."
        self.play(self.lecture[4].animate.set_color(HIGHLIGHT_COLOR))
        
        # Lemon label (Issue 30: move from B6 to C6)
        lemon_label = Text("Lemon", font_size=24, color=HIGHLIGHT_COLOR)
        self.place_at_grid(lemon_label, 'C6')
        self.play(Write(lemon_label))
        
        self.wait(2)
