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
        self.setup_layout("The Efficient Toolkit: Defining a Basis", 
            ["A basis is a space's most efficient toolkit.", 
             "It must span the entire target environment.", 
             "Every vector in it must be linearly independent.", 
             "No vector in a basis is ever redundant.", 
             "A basis perfectly describes a space's dimension."]
        )

        # === Animation for Lecture Line 1 ===
        # Display a 2D grid with two non-parallel vectors (#FFD700, #00FFFF).
        self.lecture[0].set_color(YELLOW)
        
        # Coordinate system area B2 to E5
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=3,
            y_length=3,
            background_line_style={"stroke_opacity": 0.4}
        )
        self.place_in_area(plane, 'B2', 'E5')
        
        v1 = Arrow(plane.c2p(0, 0, 0), plane.c2p(1.2, 0.4, 0), buff=0, color="#FFD700")
        v2 = Arrow(plane.c2p(0, 0, 0), plane.c2p(0.4, 1.2, 0), buff=0, color="#00FFFF")
        v1_label = MathTex("v_1", color="#FFD700", font_size=20).next_to(v1.get_end(), RIGHT, buff=0.1)
        v2_label = MathTex("v_2", color="#00FFFF", font_size=20).next_to(v2.get_end(), UP, buff=0.1)
        
        self.play(Create(plane), GrowArrow(v1), GrowArrow(v2), FadeIn(v1_label, v2_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # It must span the entire target environment.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN)
        
        # Highlight span area
        span_rect = Rectangle(
            width=3, height=3, fill_color=GREEN, fill_opacity=0.2, stroke_width=0
        ).move_to(plane.get_center())
        
        span_label = Text("Spans Floor", color="#ADFF2F", font_size=20)
        self.place_in_area(span_label, 'A2', 'A3') # Resolved Issue 25
        
        self.play(FadeIn(span_rect), FadeIn(span_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Every vector in it must be linearly independent.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN)
        
        indep_label = Text("Independent", color="#ADFF2F", font_size=20)
        self.place_at_grid(indep_label, 'A4') # Resolved Issue 25
        
        self.play(FadeIn(indep_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # No vector in a basis is ever redundant.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(RED)
        
        # Part A: One vector failure (show it only covers a line)
        one_vec_line = Line(plane.c2p(-1.8, -0.6, 0), plane.c2p(1.8, 0.6, 0), color="#FFD700", stroke_width=4)
        fail_label = Text("1 vector: Insufficient", color=WHITE, font_size=18)
        self.place_in_area(fail_label, 'F3', 'F5') # Resolved Issue 26
        
        self.play(
            FadeOut(v2, v2_label, span_rect, span_label, indep_label),
            Create(one_vec_line),
            FadeIn(fail_label)
        )
        self.wait(1.5)
        
        # Part B: Redundancy with 3rd vector u
        u = Arrow(plane.c2p(0, 0, 0), plane.c2p(1.6, 1.6, 0), buff=0, color="#FF4500")
        u_label = MathTex("u", color="#FF4500", font_size=20).next_to(u.get_end(), UR, buff=0.1)
        redundant_label = Text("Redundant", color="#FF4500", font_size=20)
        self.place_at_grid(redundant_label, 'A6')
        
        self.play(
            FadeOut(one_vec_line, fail_label),
            FadeIn(v2, v2_label, span_rect, span_label),
            GrowArrow(u),
            FadeIn(u_label, redundant_label)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        # A basis perfectly describes a space's dimension.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(GREEN)
        
        # Focus on the basis pair
        basis_circle = Circle(radius=1.0, color="#00FF00", stroke_width=2).move_to(plane.c2p(0.8, 0.8, 0))
        basis_label = Text("The Basis", color="#00FF00", font_size=24, weight=BOLD)
        self.place_in_area(basis_label, 'F4', 'F5') # Resolved Issue 27
        
        self.play(
            FadeOut(u, u_label, redundant_label, span_label),
            Create(basis_circle),
            FadeIn(basis_label)
        )
        self.wait(2)
