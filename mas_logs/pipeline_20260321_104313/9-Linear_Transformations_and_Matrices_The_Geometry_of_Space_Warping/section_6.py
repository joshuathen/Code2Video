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
        # 1. Setup Layout
        self.setup_layout(
            "Summary and Conclusion", 
            [
                "Every matrix represents a geometric dance of coordinate space.",
                "Matrices are simple maps for complex spatial transformations.",
                "These numbers describe how we warp and shape reality."
            ]
        )

        # Determine the center for the right-side animation area (B1 to F6)
        # This center will be used as the origin for spatial transformations
        tl_pos = self.grid["B1"]
        br_pos = self.grid["F6"]
        grid_center = (tl_pos + br_pos) / 2

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            background_line_style={"stroke_color": BLUE_D, "stroke_opacity": 0.5}
        )
        square = Square(side_length=1.5, color=YELLOW, fill_opacity=0.3)
        
        warp_group = VGroup(plane, square)
        # Fix for Issue 45: Position warp_group in B1-F6 area
        self.place_in_area(warp_group, "B1", "F6", scale_factor=0.7)
        
        self.play(Create(plane), Create(square))
        self.wait(0.5)

        # Transformation Sequence - Rotation
        angle = 30 * DEGREES
        rot_mat = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        self.play(
            warp_group.animate.apply_function(
                lambda p: np.dot(rot_mat, p - grid_center) + grid_center
            ),
            run_time=1.2
        )

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        map_text = Text("Geometric Maps", font_size=24, color="#00FF00")
        # Fix for Issue 44: Position map_text across A2-A5 for centering
        self.place_in_area(map_text, "A2", "A5", scale_factor=0.8)
        self.play(Write(map_text))

        # Scaling and Shearing
        scale_mat = np.array([
            [1.3, 0, 0],
            [0, 0.6, 0],
            [0, 0, 1]
        ])
        self.play(
            warp_group.animate.apply_function(
                lambda p: np.dot(scale_mat, p - grid_center) + grid_center
            ),
            run_time=1.2
        )

        shear_mat = np.array([
            [1, 0.7, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        self.play(
            warp_group.animate.apply_function(
                lambda p: np.dot(shear_mat, p - grid_center) + grid_center
            ),
            run_time=1.2
        )

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        
        # Matrix Construction (Gold)
        matrix_elements = VGroup(
            Text("a", color="#FFD700"), Text("b", color="#FFD700"),
            Text("c", color="#FFD700"), Text("d", color="#FFD700")
        ).arrange_in_grid(rows=2, cols=2, buff=0.5)
        
        l_bracket = Text("[", font_size=48, color="#FFD700").stretch_to_fit_height(matrix_elements.height + 0.2)
        l_bracket.next_to(matrix_elements, LEFT, buff=0.2)
        r_bracket = Text("]", font_size=48, color="#FFD700").stretch_to_fit_height(matrix_elements.height + 0.2)
        r_bracket.next_to(matrix_elements, RIGHT, buff=0.2)
        
        final_matrix = VGroup(matrix_elements, l_bracket, r_bracket)
        
        # Fix for Issue 43: Position final_matrix in C2-E5 area with scale 1.0
        self.place_in_area(final_matrix, "C2", "E5", scale_factor=1.0)
        
        # Final transition as per Animation Planner
        self.play(
            FadeIn(final_matrix),
            FadeOut(warp_group)
        )
        self.wait(2)
