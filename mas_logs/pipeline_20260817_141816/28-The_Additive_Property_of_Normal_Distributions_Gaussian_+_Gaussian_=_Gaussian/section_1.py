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
            "A normal distribution is defined by mean and variance.",
            "The mean represents the center of the distribution.",
            "The variance defines the spread or noise level.",
            "Visually, we represent this as a bell curve.",
            "It shows certainty centered around the mean."
        ]
        self.setup_layout("Prerequisite Warm-up: The Shape of Certainty", lecture_lines)
        
        # Define Gaussian function
        def gaussian(x, mu, sigma):
            return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

        # Create base Gaussian curve
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.5, 0.1],
            axis_config={"include_tip": False}
        ).scale(0.5)
        
        curve = axes.plot(lambda x: gaussian(x, 0, 1), color=WHITE)
        area = axes.get_area(curve, x_range=[-1, 1], color="#ADD8E6", opacity=0.3)
        
        # Add Asset: bell icon
        bell_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bell.svg").scale(0.5).set_color(WHITE)
        
        plot_group = VGroup(axes, curve, area, bell_icon)
        
        # Positioning as requested by VideoCritic (Issue 23)
        self.place_in_area(plot_group, 'B3', 'E6', scale_factor=0.65)
        
        # Positioning the bell icon on top or side of the curve
        bell_icon.next_to(axes, UP)

        # === Animation for Lecture Line 1 ===
        self.play(Create(axes), Create(curve), FadeIn(bell_icon), run_time=1.5)
        self.play(self.lecture[0].animate.set_color("#87CEEB"))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#90EE90"))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#F08080"))

        # === Animation for Lecture Line 4 ===
        self.play(FadeIn(area), self.lecture[3].animate.set_color("#87CEEB"))

        # === Animation for Lecture Line 5 ===
        self.play(curve.animate.set_color("#FFD700"), self.lecture[4].animate.set_color("#FFD700"))

        self.wait(2)
