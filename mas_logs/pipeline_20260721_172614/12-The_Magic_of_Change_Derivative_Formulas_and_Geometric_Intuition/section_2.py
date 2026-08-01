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
        # Initialize the layout with the specific title and lecture lines
        self.setup_layout(
            "Prerequisite Check: The Slope of a Line",
            [
                "The slope measures how steep a line is.",
                "It is the vertical rise over horizontal run.",
                "Constant slope means a constant rate of change."
            ]
        )
        
        # Dim all lines initially except the first
        self.lecture[1:].set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1 (already white)
        
        # Draw a white straight line (#FFFFFF) rising at a 30-degree angle
        main_line = Line(LEFT * 2, RIGHT * 2, color=WHITE).rotate(30 * DEGREES)
        self.place_in_area(main_line, 'B2', 'E5')
        
        self.play(Create(main_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2 with blue color to match triangle and dim previous
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color("#58C4DD")
        )
        
        # Blue right-angled triangle (#58C4DD) under the line
        start = main_line.get_start()
        end = main_line.get_end()
        corner = np.array([end[0], start[1], 0])
        
        dx_line = Line(start, corner, color="#58C4DD")
        dy_line = Line(corner, end, color="#58C4DD")
        triangle = VGroup(dx_line, dy_line)
        
        # Labels 'dx' and 'dy'
        dx_label = MathTex("dx", color="#58C4DD", font_size=32)
        dy_label = MathTex("dy", color="#58C4DD", font_size=32)
        
        # Resolved Issue 25: Positioning dx_label at E4
        # Resolved Issue 26: Positioning dy_label at C5
        self.place_at_grid(dx_label, 'E4', scale_factor=0.8)
        self.place_at_grid(dy_label, 'C5', scale_factor=0.8)
        
        self.play(Create(triangle))
        self.play(Write(dx_label), Write(dy_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3 with light green color to match formula and dim previous
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color("#87FF65")
        )
        
        # Flash the formula 'm = dy / dx' in light green (#87FF65)
        formula = MathTex("m = \\frac{dy}{dx}", color="#87FF65", font_size=44)
        # Resolved Issue 24: Positioning formula in area A3 to B4
        self.place_in_area(formula, 'A3', 'B4', scale_factor=1.0)
        
        self.play(Write(formula))
        self.play(Flash(formula, color="#87FF65", line_length=0.3, flash_radius=0.5))
        self.play(Indicate(formula, scale_factor=1.1, color="#87FF65"))
        
        self.wait(2)
