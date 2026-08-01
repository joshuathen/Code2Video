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
        lecture_lines = [
            "Let's apply our criteria to Pythagoras' theorem.",
            "Pip places square crackers around a triangle.",
            "Two small areas fill the largest square.",
            "Like water pouring from small tanks into one.",
            "Logic and visuals combine for a clear proof."
        ]
        self.setup_layout("Application: Explaining the Pythagorean Theorem", lecture_lines)
        
        # Geometry Definitions (3-4-5 right triangle)
        a_len = 1.2
        b_len = 1.6
        c_len = 2.0
        angle = np.arctan(a_len / b_len)
        
        # Basic triangle and squares
        triangle = Polygon(ORIGIN, UP * a_len, RIGHT * b_len, color=WHITE, stroke_width=4)
        sq_a = Square(side_length=a_len, color="#ADD8E6", stroke_width=4).next_to(triangle, LEFT, buff=0, aligned_edge=UP)
        sq_b = Square(side_length=b_len, color="#ADD8E6", stroke_width=4).next_to(triangle, DOWN, buff=0, aligned_edge=LEFT)
        
        # Hypotenuse square
        sq_c = Square(side_length=c_len, color=WHITE, stroke_width=4)
        sq_c.rotate(-angle)
        hyp_mid = (UP * a_len + RIGHT * b_len) / 2
        normal = np.array([a_len, b_len, 0]) / c_len
        sq_c.move_to(hyp_mid + normal * (c_len/2))
        
        # Visual container
        geometry = VGroup(triangle, sq_a, sq_b, sq_c)
        self.place_in_area(geometry, "B2", "E5", scale_factor=1.0)
        
        # Assets for liquid
        liquid_a = sq_a.copy().set_fill("#ADD8E6", opacity=0.7).set_stroke(width=0)
        liquid_b = sq_b.copy().set_fill("#ADD8E6", opacity=0.7).set_stroke(width=0)
        liquid_c = sq_c.copy().set_fill("#ADD8E6", opacity=0.7).set_stroke(width=0)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(triangle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(
            Create(sq_a),
            Create(sq_b),
            Create(sq_c),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(
            FadeIn(liquid_a),
            FadeIn(liquid_b),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        # Combine the small liquids and transform into the large one
        self.play(
            ReplacementTransform(VGroup(liquid_a, liquid_b), liquid_c),
            run_time=2.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Replacing MathTex with Text to avoid LaTeX dependency error
        equation = Text("a² + b² = c²", color="#FFD700")
        self.place_at_grid(equation, "F3", scale_factor=1.2)
        
        self.play(Write(equation))
        # Bright glow animation
        self.play(
            equation.animate.set_color(YELLOW).scale(1.2),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.wait(2)
        
        # Cleanup
        self.lecture[4].set_color(WHITE)
        self.wait(2)
