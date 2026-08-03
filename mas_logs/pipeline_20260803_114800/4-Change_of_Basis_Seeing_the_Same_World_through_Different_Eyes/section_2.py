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
        # Setup layout with title and lecture lines
        self.setup_layout(
            "Prerequisite Check: What is a Basis?", 
            [
                "A basis is a set of unit building blocks.", 
                "The standard basis i and j form squares.", 
                "Any two non-parallel vectors can form a basis."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Show unit vectors i=(1,0) in red (#FF0000) and j=(0,1) in green (#00FF00).
        self.lecture[0].set_color("#FF0000")
        
        # Create coordinate system elements
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": WHITE, "stroke_opacity": 0.4},
            axis_config={"include_numbers": False, "stroke_color": WHITE}
        )
        plane.set_opacity(0) # Initially hidden
        
        i_vec = Vector([1, 0], color="#FF0000")
        j_vec = Vector([0, 1], color="#00FF00")
        i_lab = MathTex(r"\hat{i}", color="#FF0000", font_size=24)
        j_lab = MathTex(r"\hat{j}", color="#00FF00", font_size=24)
        
        # Scale and position elements in the right-side area
        # Fix from Issue 32: Use A2-F6 and scale 0.6
        visual_group = VGroup(plane, i_vec, j_vec, i_lab, j_lab)
        self.place_in_area(visual_group, "A2", "F6", scale_factor=0.6)
        
        # Capture the local origin for later transformations
        local_origin = plane.get_center()
        
        # Initial positions for labels (relative to origin)
        i_lab.next_to(i_vec, DOWN, buff=0.1)
        j_lab.next_to(j_vec, LEFT, buff=0.1)
        
        # Animations
        self.play(GrowArrow(i_vec), Write(i_lab))
        self.play(GrowArrow(j_vec), Write(j_lab))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Create a full white square grid [Asset: blocks.svg] by repeating i and j vectors.
        self.lecture[0].set_color(WHITE) # Reset prev color
        self.lecture[1].set_color(WHITE)
        
        # Asset Integration (Issue 26)
        blocks_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg")
        self.place_at_grid(blocks_icon, "A6", scale_factor=0.5)
        
        self.play(
            plane.animate.set_opacity(1),
            FadeIn(blocks_icon),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Change i and j to diagonal vectors (2,1) and (-1,1) and skew grid.
        self.lecture[1].set_color(WHITE) # Keep line 2 white
        self.lecture[2].set_color("#FFFF00")
        
        # Linear transformation matrix
        matrix = [[2, -1], [1, 1]]
        
        # Since visual_group was scaled, we apply the transformation 
        # relative to the local origin.
        # We calculate the label shifts manually to keep them legible.
        scale_val = 0.6 # From Issue 32
        target_i_tip = local_origin + scale_val * np.array([2, 1, 0])
        target_j_tip = local_origin + scale_val * np.array([-1, 1, 0])
        
        self.play(
            plane.animate.apply_matrix(matrix, about_point=local_origin),
            i_vec.animate.apply_matrix(matrix, about_point=local_origin),
            j_vec.animate.apply_matrix(matrix, about_point=local_origin),
            i_lab.animate.move_to(target_i_tip + DOWN * 0.2 + RIGHT * 0.2),
            j_lab.animate.move_to(target_j_tip + UP * 0.2 + LEFT * 0.2),
            blocks_icon.animate.set_color("#FFFF00"),
            run_time=2
        )
        self.wait(2)
