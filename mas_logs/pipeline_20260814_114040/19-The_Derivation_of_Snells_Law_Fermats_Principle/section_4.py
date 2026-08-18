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
        self.setup_layout("Mathematical Derivation", [
            "Express total time as function of distance.",
            "T equals sum of times in media.",
            "Set derivative to zero for minimum time."
        ])
        
        # Colors for highlights
        c1 = "#FFFFFF"
        c2 = "#FFD700"
        c3 = "#00FF00"
        
        # === Animation for Lecture Line 1 ===
        # T(x) = (sqrt(y1^2 + x^2))/v1 + (sqrt(y2^2 + (L-x)^2))/v2
        time_eq = MathTex(
            r"T(x) = \frac{\sqrt{y_1^2 + x^2}}{v_1} + \frac{\sqrt{y_2^2 + (L-x)^2}}{v_2}", 
            font_size=28
        )
        self.place_in_area(time_eq, 'B2', 'C6', scale_factor=0.9)
        self.play(Write(time_eq))
        self.lecture[0].set_color(c1)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Differentiate T with respect to x
        dt_dx = MathTex(
            r"\frac{dT}{dx} = \frac{x}{v_1\sqrt{y_1^2 + x^2}} - \frac{L-x}{v_2\sqrt{y_2^2 + (L-x)^2}} = 0",
            font_size=26
        )
        self.place_in_area(dt_dx, 'D2', 'E6', scale_factor=0.85)
        self.play(FadeIn(dt_dx))
        self.lecture[1].set_color(c2)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Snell's Law result
        snells = MathTex(
            r"\frac{\sin\theta_1}{v_1} = \frac{\sin\theta_2}{v_2}",
            font_size=36, color=c3
        )
        self.place_at_grid(snells, 'F4', scale_factor=1.0)
        self.play(Write(snells))
        self.lecture[2].set_color(c3)
        self.wait(3)
