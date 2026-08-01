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
        # 1. SETUP LAYOUT
        title_text = "Prerequisite 1: The Nature of 'e' and 'i'"
        lecture_lines = [
            'e is the fundamental base of natural growth.', 
            'i is defined as the square root of negative one.', 
            'Multiplying by i rotates a number by ninety degrees.'
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # [Asset: e_formula]
        e_formula = MathTex("e \\approx 2.718", color="#00FF00")
        # Issue 35: Visual representation for 'e' in 'A2'-'B5'
        self.place_in_area(e_formula, 'A2', 'B5', scale_factor=0.8)
        
        # Complex Plane for visualization
        plane = ComplexPlane(
            x_range=[-4, 4, 1], 
            y_range=[-3, 3, 1], 
            background_line_style={"stroke_opacity": 0.4}
        ).add_coordinates()
        self.place_in_area(plane, 'A1', 'F6', scale_factor=0.7)
        
        # Vector growing along the x-axis
        origin = plane.n2p(0)
        e_vector_target = plane.n2p(2.718)
        vector = Arrow(origin, e_vector_target, color="#00FF00", buff=0)
        
        self.play(Write(e_formula))
        self.play(Create(plane))
        self.play(GrowArrow(vector))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.play(self.lecture[1].animate.set_color("#0000FF"))
        
        # [Asset: i_definition]
        i_definition = MathTex("i = \\sqrt{-1}", color="#0000FF")
        # Issue 36: Definition of 'i' at grid 'C4'
        self.place_at_grid(i_definition, 'C4', scale_factor=1.0)
        
        self.play(Write(i_definition))
        # Rotate vector 90 degrees CCW
        self.play(Rotate(vector, angle=PI/2, about_point=origin), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third line
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # [Asset: rotation_visual]
        rotation_visual = MathTex("i^2 = -1", color="#FFFF00")
        # Issue 37: Rotation visual in 'D2'-'F5'
        self.place_in_area(rotation_visual, 'D2', 'F5', scale_factor=0.9)
        
        self.play(Write(rotation_visual))
        # Rotate vector another 90 degrees CCW
        self.play(Rotate(vector, angle=PI/2, about_point=origin), run_time=1.5)
        self.wait(2)
