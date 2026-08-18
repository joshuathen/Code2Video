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
            "Signals are sums of harmonics.",
            "Average value is the constant term.",
            "Coefficients represent weights of frequencies.",
            "Adding waves sharpens the shape.",
            "Complexity emerges from simple components."
        ]
        self.setup_layout("The Core Formula: Breaking it Down", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Signals are sums of harmonics.
        formula = MathTex(
            "f(t) = \\frac{a_0}{2} + \\sum_{n=1}^{\\infty} \\left[ a_n \\cos(n\\omega t) + b_n \\sin(n\\omega t) \\right]",
            font_size=32
        )
        # Fix for Issue 26, 28, 37: Update placement for formula to avoid clutter and overlap
        self.place_in_area(formula, 'B2', 'C4', scale_factor=0.5)
        self.play(Write(formula))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Average value is the constant term.
        a0_box = SurroundingRectangle(formula[0][4:8], color="#FFFF00")
        self.play(Create(a0_box))
        self.lecture[1].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Coefficients represent weights of frequencies.
        an_box = SurroundingRectangle(formula[0][17:19], color="#FF00FF")
        bn_box = SurroundingRectangle(formula[0][27:29], color="#00FFFF")
        self.play(Create(an_box), Create(bn_box))
        self.lecture[2].set_color("#FF00FF")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Adding waves sharpens the shape.
        axes = Axes(x_range=[0, 2*PI, 1], y_range=[-2, 2, 1], axis_config={"include_tip": False}, x_length=4, y_length=2)
        # Fix for Issue 27, 37: Updated placement for axes
        self.place_at_grid(axes, 'D3', scale_factor=0.7)
        
        # Simple square wave approximation demo
        wave = axes.plot(lambda t: 4/PI * (np.sin(t)), color=WHITE)
        self.play(Create(axes), Create(wave))
        self.lecture[3].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Complexity emerges from simple components.
        wave_sharp = axes.plot(lambda t: 4/PI * (np.sin(t) + np.sin(3*t)/3), color="#00FF00")
        self.play(Transform(wave, wave_sharp))
        self.lecture[4].set_color("#FF8800")
        self.wait(2)
