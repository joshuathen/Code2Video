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
        # Define Colors
        COLOR_I = "#FF0000"  # Red
        COLOR_J = "#00FF00"  # Green
        COLOR_GRID = "#444444" # Dark Gray
        HIGHLIGHT_COLOR = YELLOW
        
        self.setup_layout(
            "Prerequisite: Basis Vectors as a Map",
            [
                "Every matrix transformation is defined by its columns.",
                "Columns show where unit basis vectors land.",
                "These landing spots create the map for our space."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Display a 2x2 matrix on screen with its two columns highlighted in red (#FF0000) and green (#00FF00).
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Matrix corresponding to basis landing spots (2,1) and (1,2)
        # Row 1: 2, 1
        # Row 2: 1, 2
        matrix_mobject = Matrix([[2, 1], [1, 2]])
        entries = matrix_mobject.get_entries()
        
        # Column 1 (i-hat destination): entries at index 0 and 2
        entries[0].set_color(COLOR_I)
        entries[2].set_color(COLOR_I)
        # Column 2 (j-hat destination): entries at index 1 and 3
        entries[1].set_color(COLOR_J)
        entries[3].set_color(COLOR_J)
        
        # Fix for Issue 24: Matrix overlap with grid
        self.place_at_grid(matrix_mobject, "B5", scale_factor=0.7)
        
        self.play(FadeIn(matrix_mobject))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate i-hat (#FF0000) and j-hat (#00FF00) moving from their basis positions to new coordinates (2,1) and (1,2).
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Fix for Issue 25: Plane obstructing lecture notes
        # Setup Plane in a safe area
        plane = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            background_line_style={"stroke_color": COLOR_GRID, "stroke_opacity": 0.8}
        )
        self.place_in_area(plane, "D3", "F6", scale_factor=0.5)
        plane_origin = plane.c2p(0, 0)
        
        # Initial basis vectors
        i_hat = Vector([1, 0], color=COLOR_I).shift(plane_origin)
        j_hat = Vector([0, 1], color=COLOR_J).shift(plane_origin)
        
        # Labels for vectors
        i_label = MathTex(r"\hat{i}", color=COLOR_I, font_size=24)
        j_label = MathTex(r"\hat{j}", color=COLOR_J, font_size=24)
        
        # Position labels near the initial vectors
        i_label.next_to(i_hat.get_end(), RIGHT, buff=0.1)
        j_label.next_to(j_hat.get_end(), UP, buff=0.1)
        
        self.play(FadeIn(plane), Create(i_hat), Create(j_hat), FadeIn(i_label), FadeIn(j_label))
        self.wait(0.5)
        
        # Define target coordinates in the plane's local coordinate system
        target_i_coords = plane.c2p(2, 1)
        target_j_coords = plane.c2p(1, 2)
        
        # Animate movement of basis vectors and update labels
        self.play(
            i_hat.animate.put_start_and_end_on(plane_origin, target_i_coords),
            j_hat.animate.put_start_and_end_on(plane_origin, target_j_coords),
            i_label.animate.next_to(target_i_coords, RIGHT, buff=0.1),
            j_label.animate.next_to(target_j_coords, UP, buff=0.1),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The background 2D grid (#444444) warps and stretches to align with the new positions of the basis vectors.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Create a linear transformation effect
        # The matrix from Step 1 is [[2, 1], [1, 2]]
        matrix_3x3 = np.array([[2, 1, 0], [1, 2, 0], [0, 0, 1]])
        
        def warp_func(p):
            # Transform point relative to plane origin
            local_p = p - plane_origin
            transformed_p = np.dot(matrix_3x3, local_p)
            return transformed_p + plane_origin
            
        warped_plane = plane.copy()
        warped_plane.apply_function(warp_func)
        
        # Animate the transformation of the entire grid
        self.play(
            Transform(plane, warped_plane),
            # Keep labels and vectors on top
            i_hat.animate.put_start_and_end_on(plane_origin, target_i_coords),
            j_hat.animate.put_start_and_end_on(plane_origin, target_j_coords),
            run_time=2
        )
        
        self.wait(3)
