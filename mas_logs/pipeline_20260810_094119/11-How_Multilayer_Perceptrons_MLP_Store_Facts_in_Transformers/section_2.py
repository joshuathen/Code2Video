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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "A neuron acts as a pattern matcher.",
            "It detects specific features in data.",
            "Weights determine the neuron's sensitivity."
        ]
        self.setup_layout("Prerequisite: The Concept of a Neuron as a Pattern Matcher", lecture_lines)
        
        # Mobjects
        neuron = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg", color=WHITE)
        neuron_label = Text("Neuron", font_size=20).next_to(neuron, DOWN, buff=0.1)
        neuron_group = VGroup(neuron, neuron_label)
        
        input_vector = VGroup(*[Square(side_length=0.3, color="#00FF00", fill_opacity=0.6) for _ in range(3)]).arrange(DOWN, buff=0.1)
        input_label = Text("Input", font_size=18, color="#00FF00").next_to(input_vector, LEFT)
        input_group = VGroup(input_vector, input_label)
        
        weights = Matrix([[0.5], [0.2], [0.8]], color="#0000FF").scale(0.5)
        weights_label = Text("Weights", font_size=18, color="#0000FF")
        weights_group = VGroup(weights, weights_label).arrange(DOWN)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        self.place_at_grid(input_group, "C1", scale_factor=0.8)
        self.place_at_grid(neuron_group, "C4", scale_factor=1.0)
        self.play(FadeIn(input_group), FadeIn(neuron_group))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#0000FF"))
        self.place_at_grid(weights_group, "C2", scale_factor=0.8)
        self.play(FadeIn(weights_group))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#FFFF00"))
        # Using self.place_at_grid implicitly scale again if not careful, 
        # so target color update instead of re-scaling
        self.play(neuron.animate.set_color("#FFFF00"))
        self.wait(1)
