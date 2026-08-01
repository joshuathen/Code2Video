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
        # Setup layout with mandatory lines
        self.setup_layout(
            "Defining Linear Transformation", 
            [
                "In a linear transformation, the origin stays fixed.",
                "The grid morphs while remaining flat and straight.",
                "Parallel grid lines must remain parallel and evenly spaced."
            ]
        )
        
        # Colors
        COLOR_ORIGIN = "#FFFF00"
        COLOR_VECTOR = "#FFFF00"
        COLOR_GRID = "#444444"
        COLOR_CYAN = "#00FFFF"

        # Initialize Grid
        grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={
                "stroke_color": COLOR_GRID,
                "stroke_width": 2,
                "stroke_opacity": 0.6
            },
            axis_config={"stroke_color": COLOR_GRID}
        )
        # Issue 28: Scale grid 0.8 at A1-F6
        self.place_in_area(grid, 'A1', 'F6', scale_factor=0.8)

        # Issue 30: Origin dot color #FFFF00, scale 1.2, position D4
        origin_dot = Dot(color=COLOR_ORIGIN)
        self.place_at_grid(origin_dot, 'D4', scale_factor=1.2)
        
        # Issue 29: Origin label scale 0.6, position C4
        origin_label = Text("(0,0)", font_size=20, color=WHITE)
        self.place_at_grid(origin_label, 'C4', scale_factor=0.6)

        # Vector mapping dot (starts at (1,1) relative to D4)
        vector_dot = Dot(color=COLOR_VECTOR)
        self.place_at_grid(vector_dot, 'C5', scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_ORIGIN)
        self.play(
            Create(grid),
            FadeIn(origin_dot),
            FadeIn(origin_label),
            FadeIn(vector_dot),
            run_time=2
        )
        self.wait(0.5)
        
        # Animate vector mapping: move from (1,1) [C5] to (2,-1) [E6]
        self.play(
            vector_dot.animate.move_to(self.grid['E6']),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(WHITE)
        
        # Matrix for transformation (skew)
        matrix = [[1.2, 0.6], [0.4, 1.1]]
        origin_pos = origin_dot.get_center()
        
        # Skew grid and vector dot, keeping origin fixed
        self.play(
            grid.animate.apply_matrix(matrix, about_point=origin_pos),
            vector_dot.animate.apply_matrix(matrix, about_point=origin_pos),
            FadeOut(origin_label),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_CYAN)
        
        # Highlight that grid lines remain parallel and evenly spaced
        self.play(
            grid.animate.set_color(COLOR_CYAN),
            run_time=1.5
        )
        self.wait(2)
