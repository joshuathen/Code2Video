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
            "The Gaussian is the optimal signal.",
            "It minimizes uncertainty in both domains.",
            "Rectangular windows create messy sidelobes.",
            "Gaussians decay smoothly without ringing.",
            "They are mathematically the most efficient."
        ]
        self.setup_layout("Visualizing the Gaussian Optimal", lecture_lines)
        
        # Define mobjects
        gaussian = FunctionGraph(lambda x: np.exp(-x**2), x_range=[-3, 3], color="#F5FF33")
        rect = FunctionGraph(lambda x: 1 if -1 < x < 1 else 0, x_range=[-3, 3], color="#33FFCC")
        window_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/window.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#F5FF33"))
        self.place_in_area(gaussian, "B2", "C3", scale_factor=0.6)
        self.play(Create(gaussian))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#F5FF33"))
        gauss_tracker = ValueTracker(1.0)
        # Using Updater that keeps center position consistent
        gaussian.add_updater(lambda m: m.become(FunctionGraph(
            lambda x: (1/gauss_tracker.get_value()) * np.exp(-(x/gauss_tracker.get_value())**2),
            x_range=[-3, 3], color="#F5FF33"
        ).move_to(self.grid["B2"]).scale(0.6)))
        self.play(gauss_tracker.animate.set_value(0.5), run_time=1)
        self.play(gauss_tracker.animate.set_value(1.5), run_time=1)
        gaussian.clear_updaters()
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#33FFCC"))
        self.place_in_area(rect, "B5", "C6", scale_factor=0.6)
        self.play(Create(rect))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#F5FF33"))
        self.place_at_grid(window_asset, "B4", scale_factor=0.5)
        self.play(FadeIn(window_asset))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#F5FF33"))
        label = Text("Optimal Bound", font_size=20, color=WHITE)
        self.place_at_grid(label, "D5", scale_factor=0.7)
        self.play(Write(label))
        self.wait(2)
