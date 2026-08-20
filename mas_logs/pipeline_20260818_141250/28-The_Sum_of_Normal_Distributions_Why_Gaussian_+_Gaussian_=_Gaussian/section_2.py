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
        self.setup_layout("The Rule of Summation", [
            "Two independent Gaussians sum to a Gaussian.",
            "Means are added together.",
            "Variances are also added together."
        ])
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        # (Placeholder for requested asset, as the filename suggests no actual asset)

        # Define functions for bell curves
        def bell_curve(mu, sigma):
            return FunctionGraph(lambda x: np.exp(-((x - mu)**2) / (2 * sigma**2)), x_range=[-3, 3], color=GREEN)

        # === Animation for Lecture Line 1 ===
        # Fade in two smaller distributions
        curve1 = bell_curve(-1, 0.5)
        curve2 = bell_curve(1, 0.5)
        self.place_at_grid(curve1, 'B2', scale_factor=0.6)
        self.place_at_grid(curve2, 'B5', scale_factor=0.6)
        self.play(FadeIn(curve1), FadeIn(curve2))
        self.lecture[0].set_color(GREEN)

        # === Animation for Lecture Line 2 ===
        # Slide them together
        self.play(curve1.animate.move_to(self.grid['C3']), curve2.animate.move_to(self.grid['C3']))
        self.lecture[1].set_color(GREEN)

        # === Animation for Lecture Line 3 ===
        # Morph into one larger curve
        sum_curve = bell_curve(0, 0.8)
        self.place_in_area(sum_curve, 'C3', 'C5', scale_factor=0.7)
        sum_curve.set_color(YELLOW)
        sum_label = Text("Summation", font_size=20, color="#00FFFF")
        self.place_at_grid(sum_label, 'E3', scale_factor=0.9)
        
        self.play(
            FadeOut(curve1), FadeOut(curve2),
            FadeIn(sum_curve), Write(sum_label)
        )
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
