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
        lecture_lines = [
            "The CLT bridges chaos and perfect order.",
            "Population shapes can be wild and messy.",
            "Yet, sample means aggregate into normal distributions.",
            "As samples grow, the bell curve emerges.",
            "This paradox is the magic of statistics."
        ]
        self.setup_layout("The CLT Paradox: From Chaos to Order", lecture_lines)
        
        # Dots for chaotic distribution
        dots = VGroup(*[Dot(radius=0.05, color=WHITE) for _ in range(100)])
        self.place_in_area(dots, "B2", "E5", scale_factor=0.6)
        
        # Bell curve outline
        axes = Axes(x_range=[-3, 3], y_range=[0, 1], axis_config={"include_ticks": False}).scale(0.5)
        bell = axes.plot(lambda x: np.exp(-x**2), color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"), Create(dots))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        # Random scattering within area
        for dot in dots:
            dot.generate_target()
            dot.target.move_to(self.grid["B2"] + np.random.uniform(-1, 1, 3))
        self.play(MoveToTarget(dots))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        # Grouping into normal shape
        for i, dot in enumerate(dots):
            dot.generate_target()
            x = (i / 100) * 4 - 2
            y = np.exp(-x**2)
            dot.target.move_to(axes.c2p(x, y/2))
        self.play(MoveToTarget(dots), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFF00"), Create(bell.move_to(dots.get_center())))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FFFF"))
        self.wait(1)
