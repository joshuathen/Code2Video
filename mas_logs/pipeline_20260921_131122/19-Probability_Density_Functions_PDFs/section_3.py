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
        self.setup_layout("The Meaning of Area", ["Probability is area between bounds.", "Use integrals to find area.", "Area represents total likelihood."])
        
        # Axes
        axes = Axes(x_range=[0, 6, 1], y_range=[0, 4, 1], axis_config={"include_tip": False})
        self.place_in_area(axes, "C2", "F5", scale_factor=0.5)
        
        # Bell curve-like function
        curve = axes.plot(lambda x: 3 * np.exp(-(x - 3)**2 / 2), x_range=[0, 6], color=WHITE)
        self.add(curve)
        
        a_val, b_val = 2, 4
        a_dot = Dot(axes.c2p(a_val, 0), color=WHITE)
        b_dot = Dot(axes.c2p(b_val, 0), color=WHITE)
        a_label = Text("a", font_size=20).next_to(a_dot, DOWN)
        b_label = Text("b", font_size=20).next_to(b_dot, DOWN)
        
        # Assets (Using placeholders as per instructions, though icon/none.svg is transparent)
        # Note: SVG files in assets must exist, assuming standard placeholders are handled if empty.
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"), run_time=0.5)
        self.play(FadeIn(a_dot), FadeIn(b_dot), FadeIn(a_label), FadeIn(b_label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"), run_time=0.5)
        area = axes.get_area(curve, x_range=[a_val, b_val], color="#00FF00", opacity=0.5)
        self.play(Create(area))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"), run_time=0.5)
        prob_text = MathTex(r"P(a < X < b) =", color="#FFFF00", font_size=24)
        integral_formula = MathTex(r"\int_{a}^{b} f(x) dx", color="#FFFF00", font_size=24)
        
        self.place_at_grid(prob_text, "B4", scale_factor=1.0)
        self.place_at_grid(integral_formula, "C4", scale_factor=0.9)
        
        self.play(Write(prob_text), Write(integral_formula))
        self.wait(2)
