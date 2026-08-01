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
        lecture_lines = [
            'Usually, we describe space using the standard basis.', 
            'These are unit vectors i-hat and j-hat.', 
            'Every point is just a recipe of these two vectors.'
        ]
        self.setup_layout("Prerequisite Review: The Standard Basis", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Initialize NumberPlane on the right-side grid area
        plane = NumberPlane(
            x_range=[-3, 4, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=5,
            background_line_style={"stroke_opacity": 0.4},
            axis_config={
                "include_numbers": True, 
                "font_size": 14,
                "label_constructor": Text
            }
        )
        
        # Position the plane in the designated right-side area
        self.place_in_area(plane, 'B3', 'F6', scale_factor=0.7)
        
        self.play(Create(plane))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Create standard basis vectors i-hat (red) and j-hat (green)
        i_hat = Arrow(plane.c2p(0, 0), plane.c2p(1, 0), buff=0, color="#FF0000", stroke_width=4)
        j_hat = Arrow(plane.c2p(0, 0), plane.c2p(0, 1), buff=0, color="#00FF00", stroke_width=4)
        
        # Vector labels using simple text
        i_label = Text("i", color="#FF0000", font_size=16).next_to(i_hat, DOWN, buff=0.1)
        j_label = Text("j", color="#00FF00", font_size=16).next_to(j_hat, LEFT, buff=0.1)
        
        self.play(GrowArrow(i_hat), Write(i_label))
        self.play(GrowArrow(j_hat), Write(j_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Draw vector v extending to (3, 2)
        v_coord = (3, 2)
        v_vec = Arrow(plane.c2p(0, 0), plane.c2p(*v_coord), buff=0, color="#FFFFFF", stroke_width=4)
        v_label = Text("v = [3, 2]", color="#FFFFFF", font_size=16).next_to(v_vec.get_end(), UR, buff=0.1)
        
        self.play(GrowArrow(v_vec), Write(v_label))
        self.wait(0.5)
        
        # Flash basis vectors to illustrate the 'recipe' concept
        for _ in range(2):
            self.play(
                Flash(i_hat, color="#FF0000", line_length=0.15, run_time=0.3),
                Flash(j_hat, color="#00FF00", line_length=0.15, run_time=0.3)
            )
            
        self.wait(2)
