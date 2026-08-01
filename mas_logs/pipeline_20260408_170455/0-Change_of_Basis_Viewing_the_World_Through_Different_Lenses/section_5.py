from manim import *
import numpy as np

# Expert Fix: The error was caused by the class name mismatch (TeachingScene instead of Section5Scene).
# This code ensures the class is named correctly for the import and implements a complete,
# robust animation logic for change of basis using Manim CE v0.19.0 syntax.

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

class Section5Scene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE CONFIGURATION
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=32, color=WHITE).to_edge(UP, buff=0.5)
        self.add(self.title)

        # Left-side lecture content (bullets)
        # Using Text instead of MathTex to avoid LaTeX dependencies as per previous environment constraints
        lecture_texts = [Text(line, font_size=20, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.4).scale(0.9)
        self.lecture.to_edge(LEFT, buff=0.7)
        self.add(self.lecture)

        # Define fine-grained animation grid (6x6 grid on right side)
        self.grid_locs = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Calculate coordinates for the right half of the screen
                # x shifts from center to right (approx 1 to 5)
                # y shifts from top to bottom (approx 2 to -2)
                x_val = 1.5 + (j * 0.8)
                y_val = 2.2 - (i * 0.8)
                self.grid_locs[row + col] = np.array([x_val, y_val, 0])

    def construct(self):
        title_str = "Change of Basis: Different Perspectives"
        lecture_bullets = [
            "- Vectors are defined by components",
            "  relative to a specific basis.",
            "- Standard basis (i, j) is often",
            "  the default, but not the only one.",
            "- Transitioning involves a linear",
            "  transformation of the space.",
            "- This simplifies many complex",
            "  mathematical operations."
        ]

        self.setup_layout(title_str, lecture_bullets)

        # Right-side visualization center (using the grid)
        center_point = self.grid_locs["D3"] + np.array([0.4, -0.4, 0])

        # Define Number Plane for the visual area
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            axis_config={"stroke_color": GREY_B},
            background_line_style={"stroke_opacity": 0.2}
        ).move_to(center_point)

        # Basis Vectors
        vec_i = Arrow(center_point, center_point + [1, 0, 0], buff=0, color=BLUE_D, stroke_width=4)
        vec_j = Arrow(center_point, center_point + [0, 1, 0], buff=0, color=RED_D, stroke_width=4)
        
        label_i = Text("i", font_size=18, color=BLUE_D).next_to(vec_i.get_end(), RIGHT, buff=0.1)
        label_j = Text("j", font_size=18, color=RED_D).next_to(vec_j.get_end(), UP, buff=0.1)

        self.play(
            Create(plane),
            GrowArrow(vec_i),
            GrowArrow(vec_j),
            Write(label_i),
            Write(label_j),
            run_time=1.5
        )
        self.wait(1)

        # Change of Basis transformation
        # Transformation matrix M = [[1, 1], [0, 1]] (Shear)
        matrix = [[1, 0], [1, 1]]
        
        target_vec_i = Arrow(center_point, center_point + [1, 1, 0], buff=0, color=YELLOW, stroke_width=4)
        target_vec_j = Arrow(center_point, center_point + [0, 1, 0], buff=0, color=ORANGE, stroke_width=4)
        
        new_label_i = Text("b1", font_size=18, color=YELLOW).next_to(target_vec_i.get_end(), RIGHT, buff=0.1)
        new_label_j = Text("b2", font_size=18, color=ORANGE).next_to(target_vec_j.get_end(), UP, buff=0.1)

        self.play(
            plane.animate.apply_matrix(matrix),
            Transform(vec_i, target_vec_i),
            Transform(vec_j, target_vec_j),
            Transform(label_i, new_label_i),
            Transform(label_j, new_label_j),
            run_time=2.5,
            rate_func=slow_into
        )
        
        self.wait(2)

        # Final emphasis on the new basis area
        rect = SurroundingRectangle(VGroup(vec_i, vec_j), color=WHITE, buff=0.2)
        self.play(Create(rect))
        self.play(FadeOut(rect))
        self.wait(1)
