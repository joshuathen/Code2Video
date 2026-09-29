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
            "Normal distributions follow the bell curve shape.",
            "Random variables have defined mean and variance.",
            "Independent variables means errors don't influence each other."
        ]
        self.setup_layout("Prerequisites & Intuition", lecture_lines)

        # Create assets using SVGMobject
        dot_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/dots.svg"
        group_a = SVGMobject(dot_path, color="#FF5733")
        label_a = Text("Group A", color="#33FF57", font_size=20)
        
        # === Animation for Lecture Line 1 ===
        # Create a title text 'Combinatorics Basics' in #FFFFFF at center
        title_extra = Text("Combinatorics Basics", font_size=32, color=WHITE)
        self.place_at_grid(title_extra, "A4", scale_factor=0.8)
        self.play(FadeIn(title_extra))
        self.play(self.lecture[0].animate.set_color("#FF5733"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Fade in a set of dots in #FF5733. Label as 'Group A' in #33FF57.
        self.place_in_area(group_a, "B4", "C6", scale_factor=0.7)
        self.place_at_grid(label_a, "B5", scale_factor=0.9)
        self.play(FadeIn(group_a), FadeIn(label_a))
        self.play(self.lecture[1].animate.set_color("#FF5733"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition to split screen showing two groups
        group_b = SVGMobject(dot_path, color="#FF5733")
        self.place_in_area(group_b, "E4", "F6", scale_factor=0.7)
        self.play(
            group_a.animate.set_color("#33FFBD"),
            FadeIn(group_b)
        )
        self.play(self.lecture[2].animate.set_color("#FF5733"))
        self.wait(1)
