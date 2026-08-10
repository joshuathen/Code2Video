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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Traditional models struggle with long sentences.",
            "They process information sequentially, word-by-word.",
            "Early details are often forgotten over time.",
            "We need to see the whole picture.",
            "Transformers process entire sequences simultaneously."
        ]
        self.setup_layout("The Problem: Information Overload", lecture_lines)
        
        # Assets
        brain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/brain.svg")
        eye = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/eye.svg")
        model_diag = Rectangle(width=4, height=3, color=WHITE) # Placeholder for [Asset: model_comparison_diag]
        
        # Create cluster
        dots = VGroup(*[Dot(color="#ADD8E6", radius=0.1) for _ in range(20)])
        node_cluster = VGroup(brain, dots)
        for dot in dots:
            dot.move_to(brain.get_center() + np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0]))
        
        self.place_at_grid(node_cluster, 'B5', scale_factor=0.5)
        subset = VGroup(*dots[5:10])

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(node_cluster))
        self.lecture[0].set_color("#ADD8E6")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#ADD8E6")
        self.play(Indicate(dots))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#ADD8E6")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        self.play(subset.animate.set_color("#FFFF00"), FadeIn(eye))
        self.place_at_grid(eye, 'D3', scale_factor=0.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#ADD8E6")
        self.place_in_area(model_diag, 'C3', 'F6', scale_factor=0.6)
        self.play(
            *[FadeOut(d) for d in dots if d not in subset],
            FadeIn(model_diag),
            run_time=2
        )
        self.wait(2)
