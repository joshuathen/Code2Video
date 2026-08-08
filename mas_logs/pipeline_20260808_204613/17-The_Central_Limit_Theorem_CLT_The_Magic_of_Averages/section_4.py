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
        lecture_lines = [
            "Consider delivery robots with erratic battery times.",
            "Individual depletion rates are highly skewed.",
            "The average fleet performance is predictable.",
            "We use averages to plan city logistics.",
            "CLT tames the chaos of individual robots."
        ]
        self.setup_layout("Real-World Application: The 'Robo-Delivery' Fleet", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700") # Gold
        # Create a simple visual of robots
        robots = VGroup(*[Dot(color=BLUE) for _ in range(10)])
        self.place_in_area(robots, 'A3', 'B5', scale_factor=1.0)
        self.play(FadeIn(robots))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF6347") # Tomato
        # Visualizing skewed distribution
        axes = Axes(x_range=[0, 10, 2], y_range=[0, 5, 1], axis_config={"include_tip": False}).scale(0.4)
        curve = axes.plot(lambda x: 0.2 * x**2 * np.exp(-x/1.5), color=RED)
        dist_group = VGroup(axes, curve)
        self.place_in_area(dist_group, 'C3', 'D4', scale_factor=0.7)
        self.play(Create(dist_group))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#90EE90") # LightGreen
        # Bell curve
        axes_norm = Axes(x_range=[0, 10, 2], y_range=[0, 1, 0.5], axis_config={"include_tip": False}).scale(0.4)
        curve_norm = axes_norm.plot(lambda x: np.exp(-(x-5)**2 / 2), color=GREEN)
        norm_group = VGroup(axes_norm, curve_norm)
        self.place_in_area(norm_group, 'E3', 'F4', scale_factor=0.7)
        self.play(Create(norm_group))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#87CEEB") # SkyBlue
        self.play(Flash(norm_group))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFFFF")
        self.play(FadeOut(robots), FadeOut(dist_group))
        self.play(norm_group.animate.move_to(self.grid['C3']))
