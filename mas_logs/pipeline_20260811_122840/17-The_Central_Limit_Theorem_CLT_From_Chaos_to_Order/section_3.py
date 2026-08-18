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
        self.setup_layout("The CLT Mechanism (Visual Experiment)", [
            "Pick many samples from a population.",
            "Calculate the mean for each sample.",
            "Plotting these means forms a pattern.",
            "The shape becomes a bell curve.",
            "Larger samples make the curve precise."
        ])

        # Assets
        pop_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/population.svg")
        bell_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bell.svg")
        
        # Animations setup
        # 1. Population distribution (A4-B6)
        pop_dist = pop_icon.copy()
        self.place_in_area(pop_dist, 'A4', 'B6', scale_factor=0.5)

        # 2. Bell curve (D2-E6)
        bell_curve = bell_icon.copy()
        self.place_in_area(bell_curve, 'D2', 'E6', scale_factor=0.7)
        
        # Arrow setup (directional input/output)
        arrow = Arrow(start=self.grid['B5'], end=self.grid['D5'], color=WHITE)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF5733")
        self.play(FadeIn(pop_dist))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF5733")
        self.play(GrowArrow(arrow))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#33FF57")
        self.play(FadeIn(bell_curve))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#33FF57")
        self.play(Indicate(bell_curve))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFF33")
        self.play(bell_curve.animate.set_color("#FFFF33"), run_time=1)
        self.wait(1)
