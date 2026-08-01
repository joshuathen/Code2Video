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

class Section2Scene(Scene):
    def construct(self):
        # 1. Setup Layout
        self.setup_layout(
            title_text="Matrix Exponentials: Dynamic Systems",
            lecture_lines=[
                "Linear ODE: dx/dt = Ax",
                "Matrix Exponential: e^(At)",
                "General Solution: x(t) = e^(At)x(0)",
                "Power Series: Σ (At)^k / k!"
            ]
        )

        # 2. Visual Content - Matrix and Exponential Formula
        # Using MarkupText for formatting without LaTeX dependencies
        formula = MarkupText(
            'e<sup>At</sup> = I + At + <span size="small">(At)<sup>2</sup></span>/2! + ...',
            color=YELLOW,
            font_size=32
        )
        self.place_at_grid(formula, "C4", scale_factor=0.9)

        # Manual construction of a Matrix using Text and VGroup
        matrix_a_label = Text("A = ", font_size=32)
        
        # Grid of elements
        a = Text("a", font_size=30)
        b = Text("b", font_size=30)
        c = Text("c", font_size=30)
        d = Text("d", font_size=30)
        
        matrix_elements = VGroup(
            VGroup(a, b).arrange(RIGHT, buff=0.6),
            VGroup(c, d).arrange(RIGHT, buff=0.6)
        ).arrange(DOWN, buff=0.4)
        
        # Create brackets manually using Text
        bracket_l = Text("[", font_size=60).next_to(matrix_elements, LEFT, buff=0.1)
        bracket_r = Text("]", font_size=60).next_to(matrix_elements, RIGHT, buff=0.1)
        
        matrix_vgroup = VGroup(matrix_a_label, bracket_l, matrix_elements, bracket_r).arrange(RIGHT, buff=0.1)
        self.place_at_grid(matrix_vgroup, "C2", scale_factor=1.0)

        # Animation sequence
        self.play(
            Write(matrix_vgroup),
            FadeIn(formula, shift=UP),
            run_time=2
        )
        self.wait(3)

    def setup_layout(self, title_text, lecture_lines):
        """Helper to create the base lecture layout."""
        title = Text(title_text, font_size=36, color=BLUE).to_edge(UP, buff=0.5)
        self.add(title)
        
        lecture_vgroup = VGroup(*[
            Text(f"• {line}", font_size=24) for line in lecture_lines
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        
        lecture_vgroup.to_edge(LEFT, buff=1).shift(UP * 0.5)
        self.add(lecture_vgroup)

    def place_at_grid(self, mob, pos_key, scale_factor=1.0):
        """Helper to position objects on the right side of the screen."""
        mob.scale(scale_factor)
        # Define grid coordinates for the right half of the scene
        grid_map = {
            "C2": RIGHT * 3.5 + UP * 1.2,
            "C4": RIGHT * 3.5 + DOWN * 1.2
        }
        target_pos = grid_map.get(pos_key, RIGHT * 3)
        mob.move_to(target_pos)
