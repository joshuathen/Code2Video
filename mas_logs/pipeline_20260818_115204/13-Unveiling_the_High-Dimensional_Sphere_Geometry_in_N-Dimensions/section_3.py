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
            "Does a high-dimensional sphere hold more volume?",
            "Volume approaches zero as dimensions increase.",
            "Most volume concentrates near the sphere's crust.",
            "Analogy: In high dimensions, the core is empty.",
            "Think of it as an 'Orange of Doom'."
        ]
        self.setup_layout("The Counter-Intuitive Volume Paradox", lecture_lines)
        
        # Setup visual elements
        sphere = Circle(radius=1.5, color=WHITE)
        vol_label = Text("V(n) -> 0", font_size=24)
        comparison = VGroup(Circle(radius=1.2), Circle(radius=0.4)).arrange(RIGHT, buff=0.5)
        crust_highlight = Annulus(inner_radius=1.2, outer_radius=1.5, color=RED)
        orange = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/orange.svg")
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(sphere, 'B3')
        self.play(Create(sphere), run_time=1)
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(vol_label, 'F5', scale_factor=0.9)
        self.play(FadeIn(vol_label), run_time=1)
        self.lecture[1].set_color("#FFFFFF")

        # === Animation for Lecture Line 3 ===
        self.place_in_area(comparison, 'C2', 'D4', scale_factor=0.7)
        self.play(FadeIn(comparison), run_time=1)
        self.lecture[2].set_color("#FFD700")

        # === Animation for Lecture Line 4 ===
        self.place_at_grid(crust_highlight, 'B4', scale_factor=0.6)
        self.play(Create(crust_highlight), run_time=1)
        self.lecture[3].set_color("#00FFFF")

        # === Animation for Lecture Line 5 ===
        self.place_at_grid(orange, 'C4', scale_factor=0.6)
        self.play(ReplacementTransform(sphere, orange), run_time=1)
        self.lecture[4].set_color("#FF4500")
        self.wait(2)
