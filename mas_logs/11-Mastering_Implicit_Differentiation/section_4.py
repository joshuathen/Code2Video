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
            'Differentiate each term: two x plus two y dy dx.',
            'The derivative of the constant twenty-five is zero.',
            'Group terms with dy over dx on one side.',
            'Factor out dy over dx and divide to isolate.',
            'At point (3, 4), the slope is negative three-fourths.'
        ]
        self.setup_layout("Step-by-Step Walkthrough: The Circular Track", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # Using Text instead of MathTex to avoid latex FileNotFoundError
        eq_top = Text("x² + y² = 25", font_size=30)
        self.place_in_area(eq_top, "A3", "A5", scale_factor=0.8)
        
        # Build terms for differentiation result
        d_term1 = Text("2x", color="#00FF00", font_size=30)
        d_term2 = Text("+ 2y · dy/dx", color="#00FF00", font_size=30)
        deriv_lhs = VGroup(d_term1, d_term2).arrange(RIGHT, buff=0.2)
        self.place_in_area(deriv_lhs, "B3", "B5", scale_factor=0.8)
        
        self.play(Write(eq_top))
        self.play(Write(d_term1))
        self.wait(0.5)
        self.play(Write(d_term2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FF0000")
        )
        
        # Add "= 0" to the equation
        rhs_zero = Text("= 0", color="#FF0000", font_size=30)
        rhs_zero.next_to(deriv_lhs, RIGHT, buff=0.1)
        
        self.play(Write(rhs_zero))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FF00FF")
        )
        
        # Show: 2y * dy/dx = -2x
        eq3 = VGroup(
            Text("2y · dy/dx", color=WHITE, font_size=30),
            Text("=", color=WHITE, font_size=30),
            Text("-2x", color="#FF00FF", font_size=30)
        ).arrange(RIGHT, buff=0.2)
        self.place_in_area(eq3, "C3", "C5", scale_factor=0.8)
        
        self.play(Write(eq3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#00FFFF")
        )
        
        # Resulting isolation
        eq4 = Text("dy/dx = -x/y", font_size=30)
        self.place_in_area(eq4, "D3", "D5", scale_factor=0.8)
        
        box4 = SurroundingRectangle(eq4, color="#00FFFF", buff=0.1)
        
        self.play(Write(eq4))
        self.play(Create(box4))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Substitution result text
        sub_text = Text("m = -3/4", color=YELLOW, font_size=30)
        self.place_in_area(sub_text, "E1", "E2", scale_factor=0.8)
        
        # Geometric visualization
        radius = 0.8
        circ = Circle(radius=radius, color=BLUE)
        
        # Visual point (3,4) normalized
        pt_pos = np.array([0.6 * radius, 0.8 * radius, 0])
        dot = Dot(pt_pos, color=RED, radius=0.04)
        label = Text("(3, 4)", font_size=18).next_to(dot, UR, buff=0.05)
        
        # Tangent vector
        tangent_vec = np.array([0.8, -0.6, 0])
        tangent_line = Line(
            pt_pos - tangent_vec * 0.8,
            pt_pos + tangent_vec * 0.8,
            color=YELLOW,
            stroke_width=2
        )
        
        geo_group = VGroup(circ, dot, label, tangent_line)
        self.place_in_area(geo_group, "E3", "F5", scale_factor=1.0)
        
        self.play(Write(sub_text))
        self.play(Create(circ), Create(dot), Write(label))
        self.play(Create(tangent_line))
        self.wait(2)
