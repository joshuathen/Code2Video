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
        title = "The MLP Architecture: The 'Sandwich' Structure"
        lines = [
            "MLPs use a two-step matrix multiplication process.",
            "First, the expansion layer detects specific patterns.",
            "Think of these as thousands of specialized light bulbs.",
            "If a pattern matches, the corresponding neuron fires.",
            "Then, the projection layer retrieves the associated information."
        ]
        self.setup_layout(title, lines)

        # Colors
        W1_COLOR = "#FFD700"  # Gold
        W2_COLOR = "#00BFFF"  # Deep Sky Blue
        NEURON_DIM = "#444444"
        NEURON_ACTIVE = "#FFFF00" # Bright Yellow
        HIGHLIGHT_COLOR = "#FFD700"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        # Input Vector X
        input_vector = Rectangle(height=2.0, width=0.4, color=WHITE, fill_opacity=0.3)
        input_label = Text("X", font_size=20).next_to(input_vector, UP, buff=0.1)
        input_group = VGroup(input_vector, input_label)
        self.place_at_grid(input_group, "A1", scale_factor=0.8) # Issue 36 Fix
        
        # Matrix W1
        matrix_w1 = Rectangle(height=2.0, width=1.5, color=W1_COLOR, fill_opacity=0.2)
        w1_label = MathTex("W_1", color=W1_COLOR, font_size=30).move_to(matrix_w1.get_center())
        w1_group = VGroup(matrix_w1, w1_label)
        self.place_at_grid(w1_group, "A2", scale_factor=0.8) # Issue 34 Fix
        
        self.play(FadeIn(input_group), FadeIn(w1_group))
        self.play(input_group.animate.shift(RIGHT * 1.5), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Wide Layer representation
        wide_layer = Rectangle(height=4.0, width=0.6, color=WHITE, fill_opacity=0.3)
        wide_label = Text("Expansion", font_size=18).next_to(wide_layer, DOWN, buff=0.2)
        wide_group = VGroup(wide_layer, wide_label)
        self.place_at_grid(wide_group, "B5", scale_factor=0.8) # Issue 34 Fix
        
        self.play(
            ReplacementTransform(input_group.copy(), wide_group),
            FadeOut(input_group),
            FadeOut(w1_group)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Grid of circles (4,000 "light bulbs")
        # Visualizing a 10x10 grid to represent the 4,000 neurons
        circles = VGroup(*[
            Circle(radius=0.08, color=NEURON_DIM, fill_opacity=0.8)
            for _ in range(100)
        ]).arrange_in_grid(rows=10, cols=10, buff=0.1)
        
        neuron_grid_group = VGroup(circles)
        self.place_in_area(neuron_grid_group, "B2", "E4", scale_factor=0.8) # Issue 35 Fix
        
        grid_caption = Text("4,000 Neurons", font_size=18).next_to(neuron_grid_group, UP, buff=0.2)
        
        self.play(
            ReplacementTransform(wide_group, neuron_grid_group),
            FadeIn(grid_caption)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Specific neuron #74 (Roughly middle)
        target_neuron = circles[74]
        
        # Eiffel Tower Asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/eiff.svg]
        eiff_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/eiff.svg", height=0.6, color=WHITE)
        self.place_at_grid(eiff_asset, "A3", scale_factor=1.0) # Issue 26 Fix
        
        # Pulsing glow effect
        glow = Circle(radius=0.15, color=NEURON_ACTIVE, stroke_width=0, fill_opacity=0.4)
        glow.move_to(target_neuron.get_center())
        
        def pulse_glow(mobject, dt):
            mobject.scale(1 + 0.05 * np.sin(self.time * 5))
            
        self.play(
            target_neuron.animate.set_color(NEURON_ACTIVE).set_fill(NEURON_ACTIVE, opacity=1),
            FadeIn(eiff_asset),
            FadeIn(glow)
        )
        glow.add_updater(pulse_glow)
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Matrix W2
        matrix_w2 = Rectangle(height=2.0, width=1.5, color=W2_COLOR, fill_opacity=0.2)
        w2_label = MathTex("W_2", color=W2_COLOR, font_size=30).move_to(matrix_w2.get_center())
        w2_group = VGroup(matrix_w2, w2_label)
        self.place_at_grid(w2_group, "E5", scale_factor=0.8) # Issue 35 Fix
        
        # Output vector
        output_vector = Rectangle(height=1.5, width=0.4, color=WHITE, fill_opacity=0.3)
        output_label = Text("Output", font_size=18).next_to(output_vector, DOWN, buff=0.1)
        output_group = VGroup(output_vector, output_label)
        self.place_at_grid(output_group, "E6", scale_factor=0.8) # Issue 35 Fix
        
        # Flow from grid through W2 to output
        self.play(
            FadeIn(w2_group),
            neuron_grid_group.animate.scale(0.5).move_to(self.grid["D4"]),
            glow.animate.scale(0.5).move_to(self.grid["D4"]),
            eiff_asset.animate.shift(DOWN*0.5),
            FadeOut(grid_caption)
        )
        
        self.play(
            ReplacementTransform(neuron_grid_group.copy(), output_group),
            run_time=1.5
        )
        
        self.wait(2)
        
        # Cleanup
        glow.remove_updater(pulse_glow)
        self.play(self.lecture[4].animate.set_color(WHITE))
