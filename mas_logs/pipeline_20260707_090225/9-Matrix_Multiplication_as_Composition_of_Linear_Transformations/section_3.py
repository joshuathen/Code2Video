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

class Section3Scene(TeachingScene):
    def construct(self):
        # Initial layout setup
        title_str = "Step 2: The Second Transformation (Shear)"
        lecture_lines = [
            "Now apply Matrix B, a horizontal shear device.",
            "It acts on the already rotated vector result.",
            "[Asset: Robo-Cat] is now both rotated and leaning."
        ]
        self.setup_layout(title_str, lecture_lines)
        
        # Define specific colors
        shear_color = "#FF8844"

        # === Animation for Lecture Line 1 ===
        # Create and place Matrix B formula
        # Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        matrix_b = Text(
            "B = [[1, 1], [0, 1]]",
            color=shear_color
        )
        self.place_in_area(matrix_b, 'A1', 'A3', scale_factor=0.8)
        
        self.play(
            Write(matrix_b),
            self.lecture[0].animate.set_color(shear_color)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Setup coordinate plane and Robo-Cat asset
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"color": GRAY}
        )
        
        # Load asset from specific path (Issue 53) and anchor it (Issue 42)
        robo_cat = ImageMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cat.png")
        self.place_at_grid(robo_cat, 'D3', scale_factor=0.5)
        
        # Group and place in the designated area (Issue 41)
        axes_group = Group(plane, robo_cat)
        self.place_in_area(axes_group, 'B1', 'F6', scale_factor=0.9)
        
        # Represent state from section 2: Rotate by 90 degrees CCW
        axes_group.rotate(90 * DEGREES)
        
        self.play(
            Create(plane),
            FadeIn(robo_cat),
            self.lecture[1].animate.set_color(WHITE)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Apply the horizontal shear transformation B = [[1, 1], [0, 1]]
        # This acts on the already rotated coordinate group
        shear_matrix = [[1, 1], [0, 1]]
        
        self.play(
            axes_group.animate.apply_matrix(shear_matrix),
            self.lecture[2].animate.set_color(WHITE)
        )
        self.wait(2)
