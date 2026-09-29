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
        lecture_lines = [
            "We take many samples from a population.",
            "Calculating sample means creates a new distribution.",
            "This reveals patterns hidden in the noise.",
            "Sample means act differently than individual data points.",
            "Aggregation provides structure from raw chaos."
        ]
        self.setup_layout("The Magic: Sampling and Aggregation", lecture_lines)
        
        # Color definitions for animation lines
        colors = [BLUE, GREEN, YELLOW, ORANGE, RED]
        
        # Assets
        dots_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dots.svg")
        dist_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/distribution.svg")
        
        # Placeholder Mobjects for instructions
        grid_group = VGroup()
        node_labels = Text("Labels", font_size=12)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]))
        population_dots = dots_asset.copy()
        # Per VideoCritic (Issues 26, 41)
        self.place_in_area(population_dots, 'D1', 'F6', scale_factor=0.6)
        self.play(FadeIn(population_dots))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]))
        # Per VideoCritic (Issues 27, 42, 28, 43)
        self.place_in_area(grid_group, 'A2', 'F6', scale_factor=0.8)
        self.place_at_grid(node_labels, 'D2', scale_factor=0.5)
        self.add(grid_group, node_labels)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]))
        dist = dist_asset.copy()
        self.place_in_area(dist, "A1", "C6", scale_factor=0.7)
        self.play(FadeIn(dist))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(colors[3]))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(colors[4]))
