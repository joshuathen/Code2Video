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
        # 1. Setup grid system for layout
        self.setup_grid(rows=4, cols=4)
        
        # 2. Title and Header
        title = Text("The Analytic Definition", font_size=48, color=BLUE)
        self.place_in_area(title, (0, 0), (0, 3))
        
        # 3. The Zeta Formula
        zeta_formula = Text("ζ(s) = ∑ 1/n^s", font_size=80)
        # Place in the middle area
        self.place_in_area(zeta_formula, (1, 0), (2, 3))
        
        # 4. Domain Constraint
        domain_text = Text(
            "Defined for Re(s) > 1",
            font_size=36,
            color=YELLOW
        )
        self.place_in_area(domain_text, (3, 0), (3, 3))

        # Animations
        self.play(FadeIn(title, shift=DOWN))
        self.wait(0.5)
        
        self.play(Write(zeta_formula))
        self.play(zeta_formula.animate.set_color(WHITE))
        self.wait(1)
        
        # Highlight the components
        # For Text objects, zeta_formula[0] refers to the first character 'ζ'
        rect = SurroundingRectangle(zeta_formula[0], color=BLUE, buff=0.1)
        label_zeta = Text("Riemann Zeta", font_size=24).next_to(rect, UP)
        
        self.play(Create(rect), FadeIn(label_zeta))
        self.wait(1)
        self.play(FadeOut(rect), FadeOut(label_zeta))
        
        self.play(FadeIn(domain_text, shift=UP))
        self.wait(2)
        
        # Transition to specific values
        example_s2 = Text("ζ(2) = 1 + 1/4 + 1/9 + ... = π^2/6", font_size=40)
        self.place_in_area(example_s2, (2, 0), (3, 3), scale_factor=0.9)
        
        self.play(
            domain_text.animate.shift(DOWN * 0.5).set_opacity(0.5),
            ReplacementTransform(zeta_formula.copy(), example_s2)
        )
        self.wait(3)

    def setup_grid(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid_width = config.frame_width
        self.grid_height = config.frame_height

    def place_in_area(self, mobject, start_cell, end_cell, scale_factor=1.0):
        row_start, col_start = start_cell
        row_end, col_end = end_cell
        
        cell_width = self.grid_width / self.cols
        cell_height = self.grid_height / self.rows
        
        center_x = -self.grid_width / 2 + (col_start + col_end + 1) * cell_width / 2
        center_y = self.grid_height / 2 - (row_start + row_end + 1) * cell_height / 2
        
        mobject.move_to([center_x, center_y, 0])
        mobject.scale(scale_factor)
