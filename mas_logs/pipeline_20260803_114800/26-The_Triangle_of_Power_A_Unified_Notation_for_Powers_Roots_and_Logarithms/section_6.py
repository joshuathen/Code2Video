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
        # Section Data
        title = "Operation 3: The Logarithm (Side-Climb View)"
        lecture_lines = [
            "Logarithms ask what exponent connects the base and result.",
            "We climb the side to find the top value.",
            "Ten cubed is one thousand, so the exponent is three."
        ]
        
        # Colors from Issue 33 and Storyboard
        base_color = "#3377FF"    # Blue (Issue 33)
        result_color = "#AA0000"  # Red (Issue 33)
        exp_color = "#00FF00"     # Green (Issue 33)
        climb_color = "#FFFF00"   # Yellow
        line_color = WHITE

        # Initialize Layout
        self.setup_layout(title, lecture_lines)
        
        # === Prepare Diagram ===
        # Create elements relative to diagram center
        v_base = np.array([-1.5, -1.0, 0])
        v_result = np.array([1.5, -1.0, 0])
        v_exp = np.array([0, 1.5, 0])

        triangle = Polygon(v_base, v_result, v_exp, color=line_color, stroke_width=2)
        
        # Labels for vertices
        base_val = MathTex("10", color=base_color).scale(1.2).move_to(v_base + DOWN*0.5 + LEFT*0.5)
        result_val = MathTex("1000", color=result_color).scale(1.2).move_to(v_result + DOWN*0.5 + RIGHT*0.5)
        exp_val = MathTex("3", color=exp_color).scale(1.2).move_to(v_exp + UP*0.5)
        
        # Groups for issues 32 and 33
        diagram_group = VGroup(triangle, base_val, result_val, exp_val)
        
        # Fix for Issue 32 & 33: Position the diagram in the recommended area B2-E5
        self.place_in_area(diagram_group, 'B2', 'E5', scale_factor=0.8)

        # Get the world coordinates for the climb animation after positioning
        transformed_vertices = triangle.get_vertices()
        p_base = transformed_vertices[0]
        p_result = transformed_vertices[1]
        p_exp = transformed_vertices[2]

        # === Animation for Lecture Line 1 ===
        # Highlight first line and show knowns
        self.play(self.lecture[0].animate.set_color(base_color))
        self.play(Create(triangle))
        self.play(Write(base_val), Write(result_val))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition highlight to second line and animate climb
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(climb_color)
        )
        
        # Climbing line (Yellow) along the left edge
        climb_line = Line(p_base, p_exp, color=climb_color, stroke_width=6)
        self.play(Create(climb_line), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition highlight to third line and reveal answer
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(exp_color)
        )
        
        # Reveal '3' at the top vertex with scaling and flash effect
        self.play(
            Write(exp_val),
            exp_val.animate.scale(1.5),
        )
        self.play(
            Flash(p_exp, color=exp_color, flash_radius=0.5, line_length=0.3),
            exp_val.animate.scale(1/1.5)
        )
        
        self.wait(3)
