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
        self.setup_layout("Visualizing the 'Broadening'", [
            "Adding noise increases total system uncertainty.",
            "Two narrow curves combine into a wider bell.",
            "The peak becomes flatter as variance grows."
        ])
        
        # Define functions for Gaussians
        def gaussian(x, sigma):
            return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma)**2)

        axes = Axes(
            x_range=[-4, 4, 1], y_range=[0, 1, 0.2],
            axis_config={"include_tip": False}
        )
        
        # Apply fix from issues 30/31
        self.place_in_area(axes, 'C2', 'E5', scale_factor=1.1)
        
        # Assets (none.svg placeholder)
        icon1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg").scale(0.2)
        icon2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg").scale(0.2)
        icon1.next_to(axes, UP)
        icon2.next_to(axes, DOWN)
        
        # Initial curves
        curve1 = axes.plot(lambda x: gaussian(x, 0.5), color=WHITE)
        curve2 = axes.plot(lambda x: gaussian(x, 0.7), color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(curve1), FadeIn(curve2), FadeIn(icon1), FadeIn(icon2))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Represent combined curve (e.g., sigma sqrt(0.5^2 + 0.7^2) = 0.86)
        combined_curve = axes.plot(lambda x: gaussian(x, 0.86), color=WHITE)
        self.play(
            FadeOut(curve1),
            FadeOut(curve2),
            FadeIn(combined_curve)
        )
        self.lecture[1].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(combined_curve.animate.set_color("#FF6347"), FadeOut(icon1), FadeOut(icon2))
        self.lecture[2].set_color("#FF6347")
        self.wait(2)
