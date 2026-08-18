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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Real-World Application: The AI Revolution", [
            "Convolutions are the backbone of Convolutional Neural Networks.",
            "Thousands of unique kernels recognize complex objects instantly.",
            "Digital magic brings features and patterns to life."
        ])

        # Color definitions for animations
        GRID_COLOR = "#AAAAAA"
        BOX_COLOR = "#FF0000"
        TEXT_COLOR = "#FFFFFF"
        HIGHLIGHT_COLOR = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        # Create a helper for a grid mobject
        def create_grid(rows, cols, size=1.5):
            return NumberPlane(
                x_range=[0, cols, 1],
                y_range=[0, rows, 1],
                x_length=size,
                y_length=size,
                background_line_style={"stroke_color": GRID_COLOR, "stroke_width": 2},
                axis_config={"include_ticks": False, "stroke_opacity": 0}
            )

        # Fix layout per Issue 37: grid1 ('A2'-'C4'), grid2 ('B3'-'D5'), grid3 ('C4'-'E6')
        grid1 = create_grid(4, 4)
        grid2 = create_grid(4, 4)
        grid3 = create_grid(4, 4)

        self.place_in_area(grid1, 'A2', 'C4')
        self.place_in_area(grid2, 'B3', 'D5')
        self.place_in_area(grid3, 'C4', 'E6')

        # Add connecting lines between grids to simulate network connections
        conn_lines = VGroup()
        for g_start, g_end in [(grid1, grid2), (grid2, grid3)]:
            for corner_func in [lambda m: m.get_corner(UL), lambda m: m.get_corner(UR), lambda m: m.get_corner(DL), lambda m: m.get_corner(DR)]:
                p1 = corner_func(g_start)
                p2 = corner_func(g_end)
                conn_lines.add(Line(p1, p2, color=GRID_COLOR, stroke_opacity=0.3))

        self.play(
            Create(grid1), 
            Create(grid2), 
            Create(grid3), 
            Create(conn_lines), 
            run_time=2
        )
