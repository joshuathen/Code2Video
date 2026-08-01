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
        # Setup the layout with provided lecture lines
        self.setup_layout(
            "The Rules of Linear Transformation", 
            [
                "A transformation is linear if the origin stays fixed.", 
                "Grid lines must remain straight, parallel, and evenly spaced.", 
                "This ensures space stretches without warping into curves."
            ]
        )
        
        # Visual color palette
        COLOR_1 = YELLOW
        COLOR_2 = TEAL
        COLOR_3 = "#FF69B4" 
        FLASH_COLOR = "#90EE90"
        
        # Initialize the coordinate grid mobject
        # We use a NumberPlane as the visual "grid"
        grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={
                "stroke_color": BLUE_C,
                "stroke_width": 2,
                "stroke_opacity": 0.5
            },
            axis_config={"include_tip": False, "stroke_opacity": 0.7}
        )
        # Position grid in the right side area (A1 to F6)
        # Issue 37 Fix: Scale factor reduced to 0.5
        self.place_in_area(grid, "A1", "F6", scale_factor=0.5)
        grid_center = grid.get_center()

        # Origin marker dot
        origin_dot = Dot(grid_center, radius=0.12, color=WHITE).set_z_index(10)
        origin_label = Text("Origin (0,0)", font_size=16, color=WHITE)
        # Issue 39 Fix: Better positioning for origin label
        self.place_in_area(origin_label, 'D3', 'D4', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        # "A transformation is linear if the origin stays fixed."
        self.play(
            self.lecture[0].animate.set_color(COLOR_1),
            Create(grid),
            run_time=1.5
        )
        self.play(
            FadeIn(origin_dot),
            Write(origin_label),
            run_time=0.8
        )
        
        # Demonstrating fixed origin via rotation
        self.play(
            Rotate(grid, angle=30*DEGREES, about_point=grid_center),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Grid lines must remain straight, parallel, and evenly spaced."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_2),
            FadeOut(origin_label),
            run_time=0.8
        )
        
        # Apply a shear matrix to simulate tilting while preserving parallelism
        shear_matrix = np.array([[1, 0.6, 0], [0, 1, 0], [0, 0, 1]])
        
        def linear_tilt(p):
            # Transform relative to the grid center (which is the origin)
            relative_p = p - grid_center
            new_p = np.dot(shear_matrix, relative_p)
            return new_p + grid_center

        self.play(
            grid.animate.apply_function(linear_tilt),
            run_time=2,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "This ensures space stretches without warping into curves."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_3),
            run_time=0.8
        )
        
        # Show non-linear warping (violation of linearity)
        def warp_logic(p):
            relative_p = p - grid_center
            x, y, z = relative_p
            # Add sinusoidal distortion to curve the lines
            new_x = x + 0.4 * np.sin(y * 2.0)
            new_y = y + 0.4 * np.cos(x * 2.0)
            return np.array([new_x, new_y, z]) + grid_center

        self.play(
            grid.animate.apply_function(warp_logic),
            run_time=2,
            rate_func=wiggle
        )
        
        # Create a fresh linear grid for the snap back
        clean_grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={
                "stroke_color": BLUE_C,
                "stroke_width": 2,
                "stroke_opacity": 0.5
            },
            axis_config={"include_tip": False, "stroke_opacity": 0.7}
        )
        # Issue 38 Fix: Match scale factor to 0.5
        self.place_in_area(clean_grid, "A1", "F6", scale_factor=0.5)
        
        # Visual flash effect for the "snap"
        flash_effect = Rectangle(
            width=5.2, height=5.2, 
            stroke_width=0, 
            fill_color=FLASH_COLOR, 
            fill_opacity=0.4
        ).move_to(grid_center)

        self.play(
            FadeIn(flash_effect),
            ReplacementTransform(grid, clean_grid),
            run_time=0.25
        )
        self.play(FadeOut(flash_effect), run_time=0.4)
        
        self.wait(2)
