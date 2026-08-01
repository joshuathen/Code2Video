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

class Section4InvariantsScene(TeachingScene):
    def construct(self):
        # Setup layout
        lecture_lines = [
            'Topological invariants are properties that never change.',
            'The Euler Characteristic uses vertices, edges, and faces.',
            'For any sphere-like object, the result is always two.',
            "This formula reveals a shape's hidden mathematical constant.",
            'Invariants help us distinguish between different topological spaces.'
        ]
        self.setup_layout("Topological Invariants: Euler Characteristic", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)

        # Define Cube (Wireframe)
        cube_front = Square(side_length=2, color="#00BFFF")
        cube_back = Square(side_length=2, color="#00BFFF").shift(RIGHT*0.5 + UP*0.5)
        cube_edges = VGroup(*[
            Line(cube_front.get_corner(corner), cube_back.get_corner(corner), color="#00BFFF")
            for corner in [UL, UR, DL, DR]
        ])
        cube = VGroup(cube_front, cube_back, cube_edges)
        self.place_in_area(cube, "B1", "D4", scale_factor=0.6)
        
        # Cube Labels
        vef_label = Text("V=8, E=12, F=6", font_size=24, color="#FFFFFF")
        self.place_at_grid(vef_label, "B5", scale_factor=1.0)
        
        self.play(Create(cube), FadeIn(vef_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(BLUE)

        # Calculation Display
        chi_calc = Text("8 - 12 + 6 = 2", font_size=32, color="#00FF00")
        chi_result = Text("Chi = 2", font_size=36, color="#00FF00")
        self.place_at_grid(chi_calc, "D5", scale_factor=1.0)
        self.place_at_grid(chi_result, "C5", scale_factor=1.2)

        self.play(Write(chi_calc))
        self.wait(0.5)
        self.play(FadeIn(chi_result))
        
        # Morph cube to sphere (Circle in 2D representation)
        sphere = Circle(radius=1.2, color="#00BFFF", fill_opacity=0.3)
        self.place_in_area(sphere, "B1", "D4", scale_factor=0.6)
        
        self.play(
            ReplacementTransform(cube, sphere),
            chi_result.animate.set_color("#00FF00"), # Keep static/highlighted
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(BLUE)

        # Define Tetrahedron
        # p1=bottom-left, p2=bottom-right, p3=inner-depth, p4=top
        p1 = [-1, -0.5, 0]
        p2 = [1, -0.5, 0]
        p3 = [0.2, 0, 0]
        p4 = [0, 1.2, 0]
        tetra_edges = VGroup(
            Line(p1, p2), Line(p2, p3), Line(p3, p1),
            Line(p1, p4), Line(p2, p4), Line(p3, p4)
        ).set_color("#00BFFF")
        tetrahedron = VGroup(tetra_edges)
        self.place_in_area(tetrahedron, "B1", "D4", scale_factor=0.8)
        
        tetra_calc = Text("4 - 6 + 4 = 2", font_size=32, color="#00FF00")
        self.place_at_grid(tetra_calc, "E5", scale_factor=1.0)

        self.play(
            FadeOut(sphere),
            FadeOut(vef_label),
            FadeIn(tetrahedron)
        )
        self.play(Write(tetra_calc))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(BLUE)

        # Highlight '2' results
        highlight1 = SurroundingRectangle(chi_calc[-1], color=WHITE, buff=0.1)
        highlight2 = SurroundingRectangle(tetra_calc[-1], color=WHITE, buff=0.1)
        highlight3 = SurroundingRectangle(chi_result[-1], color=WHITE, buff=0.1)

        self.play(
            Create(highlight1),
            Create(highlight2),
            Create(highlight3)
        )
        self.wait(2)
