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
        # Initial Setup
        title_str = "Prerequisite Knowledge: Powers of Sine"
        lecture_lines = [
            "- Let I sub n be the integral of sine powers.",
            "- Increasing the exponent n shrinks the area under sine.",
            "- We examine this area from zero to Pi halves."
        ]
        self.setup_layout(title_str, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Replacement: Using Text instead of MathTex to avoid FileNotFoundError: 'latex'
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Integral formula as Text to avoid LaTeX dependency
        integral_formula = Text(
            "I_n = ∫ sin^n(x) dx from 0 to π/2",
            font_size=24,
            color=WHITE
        )
        self.place_in_area(integral_formula, "A2", "A5", scale_factor=0.9)
        
        self.play(Write(integral_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # Axes setup
        axes = Axes(
            x_range=[0, PI/2 + 0.3, PI/4],
            y_range=[0, 1.2, 0.5],
            x_length=3.5,
            y_length=2.5,
            axis_config={"include_tip": True, "color": GREY_C}
        )
        # Using Text mobjects for labels to avoid internal MathTex calls in get_axis_labels
        axes_labels = axes.get_axis_labels(
            x_label=Text("x", font_size=18), 
            y_label=Text("y", font_size=18)
        ).scale(0.7)
        
        axes_group = VGroup(axes, axes_labels)
        self.place_in_area(axes_group, "C2", "E5", scale_factor=0.9)
        
        # Sine Curves
        sin1 = axes.plot(lambda x: np.sin(x), x_range=[0, PI/2], color="#0000FF")
        sin10 = axes.plot(lambda x: np.sin(x)**10, x_range=[0, PI/2], color="#00FF00")
        
        # Areas
        area1 = axes.get_area(sin1, x_range=[0, PI/2], color="#0000FF", opacity=0.2)
        area10 = axes.get_area(sin10, x_range=[0, PI/2], color="#00FF00", opacity=0.4)
        
        self.play(Create(axes), FadeIn(axes_labels))
        self.play(Create(sin1), FadeIn(area1))
        self.wait(0.5)
        self.play(Create(sin10), FadeIn(area10))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # Inequality as Text to avoid LaTeX dependency
        inequality = Text("I_n < I_{n-1}", font_size=24, color="#FFFF00")
        self.place_at_grid(inequality, "D6", scale_factor=1.0)
        
        # Pi/2 marker as Text to avoid LaTeX dependency
        pi_half_marker = Text("π/2", font_size=18).move_to(axes.c2p(PI/2, -0.3))
        
        self.play(Write(inequality))
        self.play(FadeIn(pi_half_marker))
        
        self.wait(4)

        # Cleanup
        self.play(
            FadeOut(self.lecture),
            FadeOut(self.title),
            FadeOut(integral_formula),
            FadeOut(axes_group),
            FadeOut(sin1),
            FadeOut(sin10),
            FadeOut(area1),
            FadeOut(area10),
            FadeOut(inequality),
            FadeOut(pi_half_marker)
        )
