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
        title = "The Second Layer: Fact Retrieval (Values)"
        lines = [
            "The second layer retrieves the corresponding values.",
            "Activating a key releases its stored data vector.",
            "This adds the specific fact to the model's state."
        ]
        self.setup_layout(title, lines)

        # Colors
        NEURON_COLOR = "#D3D3D3"
        MATRIX_COLOR = "#D3D3D3"
        GLOW_COLOR = "#FFFF00"
        VECTOR_COLOR = "#00FF00"

        # === Animation for Lecture Line 1 ===
        # Show matrix W2 connected to the previous neuron grid.
        self.lecture[0].set_color(MATRIX_COLOR)
        
        # Neuron Grid (First Layer / Keys) - Area B1 to D3 (Issue 40 Fix)
        neuron_grid = VGroup(*[
            Square(side_length=0.3, color=NEURON_COLOR, stroke_width=2) 
            for _ in range(9)
        ]).arrange_in_grid(rows=3, cols=3, buff=0.15)
        self.place_in_area(neuron_grid, "B1", "D3")
        
        neuron_label = Text("Key Neurons", font_size=18, color=NEURON_COLOR)
        self.place_at_grid(neuron_label, "A2", scale_factor=0.8)

        # W2 Matrix (Second Layer / Values) - Area B4 to E6 (Issue 40 Fix)
        w2_matrix = VGroup(*[
            Square(side_length=0.3, color=MATRIX_COLOR, stroke_width=2) 
            for _ in range(12)
        ]).arrange_in_grid(rows=4, cols=3, buff=0.15)
        self.place_in_area(w2_matrix, "B4", "E6")
        
        w2_label = MathTex("W_2", font_size=24, color=MATRIX_COLOR)
        self.place_at_grid(w2_label, "A5", scale_factor=1.0)

        # Connecting lines (visual representative)
        connections = VGroup(*[
            Line(neuron_grid.get_right(), w2_matrix.get_left(), stroke_width=1, color=NEURON_COLOR, stroke_opacity=0.3)
            for _ in range(3)
        ]).arrange(DOWN, buff=0.5)

        self.play(
            Create(neuron_grid),
            Write(neuron_label),
            FadeIn(w2_matrix),
            Write(w2_label),
            Create(connections)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Glowing neuron #402 sends a signal into the W2 matrix.
        self.lecture[1].set_color(GLOW_COLOR)
        
        # Pick one neuron to be "Neuron #402"
        active_neuron = neuron_grid[4] # Center one
        active_neuron_glow = active_neuron.copy().set_fill(GLOW_COLOR, opacity=0.5).set_color(GLOW_COLOR)
        
        neuron_402_label = Text("#402", font_size=16, color=GLOW_COLOR)
        self.place_at_grid(neuron_402_label, "E2", scale_factor=0.8) # Issue 42 Fix

        # Signal pulse
        signal = Dot(color=GLOW_COLOR).move_to(active_neuron.get_center())
        
        # Target in W2 (representing the value vector associated with neuron 402)
        target_row_in_w2 = VGroup(*w2_matrix[3:6]) # Row 2 of W2

        self.play(
            active_neuron.animate.set_color(GLOW_COLOR),
            FadeIn(active_neuron_glow),
            Write(neuron_402_label)
        )
        
        self.play(
            signal.animate.move_to(target_row_in_w2.get_center()),
            target_row_in_w2.animate.set_color(GLOW_COLOR),
            run_time=1.5
        )
        self.play(FadeOut(signal))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A vector labeled 'Paris' [Asset: paris.svg] emerges from W2 and merges into the stream.
        self.lecture[2].set_color(VECTOR_COLOR)
        
        # "Paris" Vector Asset Integration (Issue 32)
        paris_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/paris.svg")
        paris_icon.set_color(VECTOR_COLOR)
        paris_text = Text("Paris", font_size=16, color=VECTOR_COLOR)
        paris_asset = VGroup(paris_icon, paris_text).arrange(DOWN, buff=0.1)
        paris_asset.scale(0.5).move_to(target_row_in_w2.get_center())
        
        # Hidden State Stream at the bottom (F1 to F6)
        stream_line = Line(self.grid["F1"], self.grid["F6"], color=WHITE, stroke_width=2)
        stream_label = Text("Hidden State Stream", font_size=16, color=WHITE)
        self.place_in_area(stream_label, "F4", "F6", scale_factor=0.8) # Issue 41 Fix
        
        # Initial hidden state particles
        stream_particles = VGroup(*[
            Dot(radius=0.05, color=BLUE_B).move_to(self.grid["F" + str(i)])
            for i in range(1, 4)
        ])

        self.play(
            Create(stream_line),
            Write(stream_label),
            FadeIn(stream_particles)
        )
        
        # Paris vector emerges and merges
        merge_point = self.grid["F5"]
        self.play(
            paris_asset.animate.move_to(merge_point).scale(0.8),
            run_time=2
        )
        
        # Interaction at stream: "cloud of Paris-related particles"
        # Seed the random generator for deterministic output if needed, though Manim usually handles it.
        particles = VGroup(*[
            Dot(radius=0.03, color=VECTOR_COLOR).move_to(merge_point + np.array([np.random.uniform(-0.3, 0.3), np.random.uniform(-0.2, 0.2), 0]))
            for _ in range(10)
        ])
        
        self.play(
            FadeOut(paris_asset),
            FadeIn(particles)
        )
        self.play(
            particles.animate.shift(RIGHT * 1.5).set_opacity(0),
            rate_func=linear,
            run_time=2
        )
        
        self.wait(2)
