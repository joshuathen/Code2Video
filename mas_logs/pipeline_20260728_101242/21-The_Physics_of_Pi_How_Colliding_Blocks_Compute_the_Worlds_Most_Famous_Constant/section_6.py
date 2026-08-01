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

class Section6Scene(TeachingScene):
    def construct(self):
        title_str = "The Final Connection: Why Pi?"
        lecture_lines_str = [
            "The mass ratio determines each reflection's small angle.",
            "As M grows, the angle theta becomes tiny.",
            "Total collisions equal Pi radians divided by theta.",
            "For powers of 100, we recover Pi's digits.",
            "Physics has just computed Pi through simple collisions."
        ]
        self.setup_layout(title_str, lecture_lines_str)
        
        # Colors
        COLOR_ARC = "#FF8C00"
        COLOR_FORMULA = "#FFFFFF"
        COLOR_SEMI = "#FFD700"
        COLOR_PI = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_ARC)
        
        # Base circle
        circle = Circle(radius=1.5, color=GREY_D, stroke_width=2)
        self.place_in_area(circle, "B2", "E5")
        
        # A small arc to show theta
        theta_val = 0.4
        small_arc = Arc(radius=1.5, start_angle=PI, angle=-theta_val, color=COLOR_ARC, stroke_width=6)
        small_arc.move_to(circle.get_center())
        
        label_theta = MathTex(r"\theta", color=COLOR_ARC)
        self.place_at_grid(label_theta, "C1", scale_factor=1.2)

        self.play(Create(circle), run_time=1)
        self.play(Create(small_arc), Write(label_theta), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_FORMULA)
        
        formula_theta = MathTex(r"\theta \approx \sqrt{\frac{m}{M}}", color=COLOR_FORMULA)
        # Fix: Issue 32/34 - Use place_in_area for better formula layout
        self.place_in_area(formula_theta, "A1", "A3", scale_factor=0.8)
        
        self.play(Write(formula_theta))
        
        # Simulate M growing
        new_theta_val = 0.1
        smaller_arc = Arc(radius=1.5, start_angle=PI, angle=-new_theta_val, color=COLOR_ARC, stroke_width=6)
        smaller_arc.move_to(circle.get_center())
        
        self.play(Transform(small_arc, smaller_arc), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_SEMI)
        
        # Show semi-circle path
        semi_circle = Arc(radius=1.5, start_angle=PI, angle=-PI, color=COLOR_SEMI, stroke_width=4)
        semi_circle.move_to(circle.get_center())
        
        label_pi = MathTex(r"\pi \text{ rad}", color=COLOR_SEMI)
        self.place_at_grid(label_pi, "F3", scale_factor=0.8)
        
        self.play(Create(semi_circle), Write(label_pi), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_FORMULA)
        
        formula_n = MathTex(r"N = \left\lfloor \frac{\pi}{\theta} \right\rfloor", color=COLOR_FORMULA)
        # Fix: Issue 32/35 - Use place_in_area for better formula layout
        self.place_in_area(formula_n, "A4", "A6", scale_factor=0.8)
        
        self.play(Write(formula_n))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_PI)
        
        pi_digits = Text("N = 314159...", color=COLOR_PI, font_size=36)
        # Fix: Issue 33/36 - Use place_in_area for long string
        self.place_in_area(pi_digits, "F4", "F6", scale_factor=1.0)
        
        self.play(FadeIn(pi_digits, shift=UP))
        self.wait(2)
