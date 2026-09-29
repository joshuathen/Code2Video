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
            "Continuous variables have zero point probability.",
            "We measure probability over an interval instead.",
            "Area under the curve equals the probability.",
            "Integral calculates this area between two points.",
            "Visualize probability as a shaded curve slice."
        ]
        self.setup_layout("Interpreting Probability as Area", lecture_lines)
        
        # Assets placeholders
        # NOTE: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg is a dummy
        
        # Setup Axes and Curve
        axes = Axes(x_range=[-3, 3, 1], y_range=[0, 1.5, 0.5], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: np.exp(-x**2), x_range=[-3, 3], color=WHITE)
        plot_group = VGroup(axes, curve)
        
        # Apply positioning constraints from VideoCritic
        # Fix 22, 23, 24: Use B3, F5, scale 0.55 for better vertical/horizontal alignment
        self.place_in_area(plot_group, 'B3', 'F5', scale_factor=0.55)

        # === Animation for Lecture Line 1 ===
        self.play(Create(plot_group))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF7F50")
        interval_start = -0.5
        interval_end = 0.5
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#87CEFA")
        area = axes.get_area(curve, x_range=[interval_start, interval_end], color="#87CEFA", opacity=0.5)
        self.play(FadeIn(area))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFD700")
        integral_label = MathTex(r"\int_{a}^{b} f(x) dx", color="#FFD700").scale(0.8)
        self.place_at_grid(integral_label, 'A4')
        self.play(Write(integral_label))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF0000")
        dot = Dot(point=axes.c2p(0, np.exp(0), 0), color=RED)
        point_label = Text("P(X=x) = 0", font_size=16, color=RED)
        self.place_at_grid(dot, 'B5') # dummy location
        self.play(FadeIn(dot), Write(point_label.next_to(dot, RIGHT)))
        self.wait(2)
