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

class Section5Scene(TeachingScene):
    def construct(self):
        title = "Layered Intelligence: Structure"
        lines = [
            "Neurons are organized into input, hidden, and output layers.",
            "Hidden layers extract increasingly complex features from the data.",
            "Information flows from left to right through the network."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_INPUT = "#B0C4DE"
        COLOR_HIDDEN = "#FFA500"
        COLOR_OUTPUT = "#90EE90"
        COLOR_PULSE = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(COLOR_INPUT))

        # Create Layers
        # Input Layer (3 nodes)
        input_nodes = VGroup(*[Circle(radius=0.15, color=COLOR_INPUT, fill_opacity=0.8, fill_color=COLOR_INPUT) for _ in range(3)])
        self.place_at_grid(input_nodes[0], "B2")
        self.place_at_grid(input_nodes[1], "C2")
        self.place_at_grid(input_nodes[2], "D2")

        # Hidden Layer (4 nodes)
        hidden_nodes = VGroup(*[Circle(radius=0.15, color=COLOR_HIDDEN, fill_opacity=0.8, fill_color=COLOR_HIDDEN) for _ in range(4)])
        self.place_at_grid(hidden_nodes[0], "B4")
        self.place_at_grid(hidden_nodes[1], "C4")
        self.place_at_grid(hidden_nodes[2], "D4")
        self.place_at_grid(hidden_nodes[3], "E4")

        # Output Layer (2 nodes)
        output_nodes = VGroup(*[Circle(radius=0.15, color=COLOR_OUTPUT, fill_opacity=0.8, fill_color=COLOR_OUTPUT) for _ in range(2)])
        self.place_at_grid(output_nodes[0], "C6")
        self.place_at_grid(output_nodes[1], "D6")

        # Labels
        input_label = Text("Sensors", font_size=18, color=COLOR_INPUT)
        self.place_at_grid(input_label, "A2")
        
        hidden_label = Text("Feature Extractors", font_size=18, color=COLOR_HIDDEN)
        # Fix for Issue 33: Use area A3-A5, scale 0.7 to avoid crowding
        self.place_in_area(hidden_label, "A3", "A5", scale_factor=0.7)

        output_label = Text("Decision", font_size=18, color=COLOR_OUTPUT)
        # Fix for Issue 34: B6, scale 0.8 to avoid boundary issues
        self.place_at_grid(output_label, "B6", scale_factor=0.8)

        self.play(
            LaggedStart(
                Create(input_nodes),
                Write(input_label),
                Create(hidden_nodes),
                Write(hidden_label),
                Create(output_nodes),
                Write(output_label),
                lag_ratio=0.3
            )
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_HIDDEN)
        )

        # Complexity Visuals
        edge_text = Text("Edges", font_size=14, color=WHITE)
        circle_text = Text("Circles", font_size=14, color=WHITE)
        
        # Asset for Issue 20: Cat Eye [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png]
        cat_eye_asset = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")

        # Positioning icons near hidden nodes
        self.place_at_grid(edge_text, "B3", scale_factor=0.8)
        self.place_at_grid(circle_text, "C3", scale_factor=0.8)
        
        # Fix for Issue 35: area D3-D3, scale 0.7 for the Cat Eye icon
        self.place_in_area(cat_eye_asset, "D3", "D3", scale_factor=0.7)

        # Animate the appearance of features
        self.play(FadeIn(edge_text, shift=UP*0.2))
        self.wait(0.5)
        self.play(ReplacementTransform(edge_text, circle_text))
        self.wait(0.5)
        self.play(FadeOut(circle_text), FadeIn(cat_eye_asset))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_PULSE)
        )

        # Connections
        connections = VGroup()
        for in_node in input_nodes:
            for hid_node in hidden_nodes:
                line = Line(in_node.get_right(), hid_node.get_left(), stroke_width=1, color=GRAY_C, stroke_opacity=0.5)
                connections.add(line)
        
        for hid_node in hidden_nodes:
            for out_node in output_nodes:
                line = Line(hid_node.get_right(), out_node.get_left(), stroke_width=1, color=GRAY_C, stroke_opacity=0.5)
                connections.add(line)

        self.play(Create(connections), run_time=2)

        # Pulses
        def get_pulse_animation(path_group):
            animations = []
            for path in path_group:
                dot = Dot(color=COLOR_PULSE, radius=0.04)
                animations.append(MoveAlongPath(dot, path, rate_func=linear))
            return animations

        # Information flow pulses (Optimized: dots are removed after animation)
        n_in = len(input_nodes)
        n_hid = len(hidden_nodes)
        
        pulses_in_to_hid = get_pulse_animation(connections[:n_in*n_hid])
        dots_1 = [anim.mobject for anim in pulses_in_to_hid]
        self.play(AnimationGroup(*pulses_in_to_hid), run_time=1.5)
        self.remove(*dots_1)
        
        pulses_hid_to_out = get_pulse_animation(connections[n_in*n_hid:])
        dots_2 = [anim.mobject for anim in pulses_hid_to_out]
        self.play(AnimationGroup(*pulses_hid_to_out), run_time=1.5)
        self.remove(*dots_2)

        self.wait(2)
        
        # Cleanup colors for the final state of the scene
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
