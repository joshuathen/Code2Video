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

class Section1Scene(TeachingScene):
    def construct(self):
        # Data from shared state
        title_text = "The Perspective Problem"
        lecture_lines = [
            "One point can have different coordinates in different grids.",
            "We describe its position using standard floor tiles.",
            "The robot uses its own tilted, custom grid."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_STD = WHITE
        COLOR_SLANTED = "#5271FF"
        HIGHLIGHT_COLOR = YELLOW
        
        # === Animation for Lecture Line 1 ===
        # L009: Scan for [Asset: filename]
        # [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/robot.svg]
        
        std_grid = NumberPlane(
            x_range=[-4, 4], 
            y_range=[-4, 4], 
            background_line_style={"stroke_color": COLOR_STD, "stroke_opacity": 0.2},
            axis_config={"stroke_color": COLOR_STD, "stroke_width": 1}
        )
        # Issue 24: Fix std_grid position to avoid crowding title and lecture
        self.place_in_area(std_grid, "C3", "F6", scale_factor=0.45)
        
        # Load robot asset
        robot = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/robot.svg")
        robot.set_color(WHITE)
        
        pos_z4 = std_grid.c2p(2, 1)
        robot.scale(0.3).move_to(pos_z4 + LEFT * 0.4)
        
        z4_dot = Dot(pos_z4, color=COLOR_STD)
        z4_label = Text("Z-4", font_size=20, color=COLOR_STD)
        z4_label.next_to(robot, UP, buff=0.1)
        
        self.play(
            Create(std_grid),
            FadeIn(robot),
            FadeIn(z4_dot),
            Write(z4_label),
            self.lecture[0].animate.set_color(HIGHLIGHT_COLOR),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "We describe its position using standard floor tiles."
        # Highlight standard coordinates (2,1)
        
        h_line = std_grid.get_horizontal_line(pos_z4, color=COLOR_STD, stroke_width=1)
        v_line = std_grid.get_vertical_line(pos_z4, color=COLOR_STD, stroke_width=1)
        
        std_coords = Text("(2, 1) std", color=COLOR_STD, font_size=20)
        # Issue 25: Place coordinate labels at specific grid points to prevent clutter
        self.place_at_grid(std_coords, "D4", scale_factor=0.8)
        
        self.play(
            Create(h_line),
            Create(v_line),
            Write(std_coords),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The robot uses its own tilted, custom grid."
        # Fade in slanted blue grid and show slanted coordinate (1,0)
        
        slanted_grid = NumberPlane(
            x_range=[-4, 4], 
            y_range=[-4, 4], 
            background_line_style={"stroke_color": COLOR_SLANTED, "stroke_opacity": 0.4},
            axis_config={"stroke_color": COLOR_SLANTED, "stroke_width": 2}
        )
        # Apply transformation matrix mapping standard basis to basis {(2,1), (-1,1)}
        slanted_grid.apply_matrix([[2, -1], [1, 1]])
        # Issue 23: Fix slanted_grid position to avoid obstructing lecture notes
        self.place_in_area(slanted_grid, "C3", "F6", scale_factor=0.45)
        
        # Slanted highlight (vector along first basis direction)
        slanted_vec = Arrow(
            std_grid.get_origin(), 
            pos_z4, 
            buff=0, 
            color=COLOR_SLANTED, 
            stroke_width=4
        )
        
        slanted_coords = Text("(1, 0) blue", color=COLOR_SLANTED, font_size=20)
        # Issue 25: Place coordinate labels at specific grid points
        self.place_at_grid(slanted_coords, "D5", scale_factor=0.8)
        
        self.play(
            Create(slanted_grid),
            GrowArrow(slanted_vec),
            Write(slanted_coords),
            self.lecture[2].animate.set_color(COLOR_SLANTED),
            run_time=2
        )
        self.wait(3)
