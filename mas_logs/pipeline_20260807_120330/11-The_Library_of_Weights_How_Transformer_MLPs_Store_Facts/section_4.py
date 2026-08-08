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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "The First Layer: The Pattern Detectors (Keys)",
            [
                "The first layer consists of concept detectors.",
                "Each neuron represents a specific pattern or key.",
                "These keys identify what the input is about."
            ]
        )

        # Colors
        GRAY_COLOR = "#D3D3D3"
        INPUT_COLOR = "#ADD8E6"
        GLOW_COLOR = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Show a matrix W1 and a grid representing 1000 neurons. Color: #D3D3D3.
        self.lecture[0].set_color(GRAY_COLOR)
        
        w1_label = MathTex("W_1", color=GRAY_COLOR)
        self.place_in_area(w1_label, 'A4', 'A5', scale_factor=0.8) # Fixed Issue 37
        
        # Representing 1000 neurons with an 8x8 grid of dots for visual clarity
        neuron_grid = VGroup(*[
            Circle(radius=0.08, color=GRAY_COLOR, fill_opacity=0.3)
            for _ in range(64)
        ]).arrange_in_grid(rows=8, cols=8, buff=0.15)
        
        self.place_in_area(neuron_grid, "B3", "E6", scale_factor=1.0)
        
        self.play(
            FadeIn(w1_label),
            Create(neuron_grid),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Input 'Eiffel Tower' flows into the neuron grid. Color: #ADD8E6.
        self.lecture[1].set_color(INPUT_COLOR)
        
        input_text = Text("Eiffel Tower", font_size=20, color=INPUT_COLOR)
        # Starting point for input
        self.place_at_grid(input_text, 'B2', scale_factor=1.0) # Fixed Issue 38
        
        # Path for the input text to flow into the grid
        target_pos = self.grid["B3"]
        
        self.play(
            FadeIn(input_text),
            run_time=0.5
        )
        self.play(
            input_text.animate.move_to(target_pos),
            run_time=1.5,
            rate_func=slow_into
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Neuron #402 ('French Landmarks') glows brightly while others stay dim. Color: #FFFF00.
        self.lecture[2].set_color(GLOW_COLOR)
        
        # Pick a specific neuron to represent #402 (e.g., the one at index 27)
        target_neuron_index = 27
        target_neuron = neuron_grid[target_neuron_index]
        
        neuron_label = Text("French Landmarks", font_size=18, color=GLOW_COLOR)
        # Position label near the target neuron. 
        self.place_in_area(neuron_label, 'F4', 'F5', scale_factor=0.9) # Fixed Issue 39
        
        # Highlight animation
        glow_circle = Circle(radius=0.12, color=GLOW_COLOR, fill_opacity=0.8)
        glow_circle.move_to(target_neuron.get_center())
        
        # Connect input to the specific neuron with a line
        flow_line = Line(
            input_text.get_right(), 
            glow_circle.get_center(), 
            color=INPUT_COLOR, 
            stroke_width=2
        )

        self.play(
            Create(flow_line),
            target_neuron.animate.set_color(GLOW_COLOR).set_fill(GLOW_COLOR, opacity=1.0),
            FadeIn(glow_circle, scale=1.2),
            FadeIn(neuron_label),
            run_time=1.5
        )
        
        # Pulsing effect for the "glow"
        self.play(
            glow_circle.animate.scale(1.3),
            rate_func=there_and_back,
            run_time=1
        )
        
        self.wait(3)
