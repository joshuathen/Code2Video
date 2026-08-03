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
        # Setup layout
        title_text = "The Concept of Perspective"
        lecture_lines = [
            "Meet Pixel the Robot standing in a room.",
            "We see Pixel at coordinates (3, 2).",
            "But from Pixel's view, his position feels different."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_STANDARD = WHITE
        COLOR_POINT = "#00FFFF"
        COLOR_TILTED = "#FFFF00"
        
        # Asset path
        ROBOT_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg"

        # === Animation for Lecture Line 1 ===
        # "Meet Pixel the Robot standing in a room."
        self.lecture[0].set_color(COLOR_POINT)
        
        # Standard Grid
        # Origin at E2 (x=1.5, y=-1.8)
        standard_grid = NumberPlane(
            x_range=[-1, 4, 1],
            y_range=[-1, 4, 1],
            background_line_style={"stroke_color": GREY_D, "stroke_opacity": 0.5},
            axis_config={"stroke_color": COLOR_STANDARD, "include_tip": True}
        )
        # Position origin (0,0) of NumberPlane at grid point E2
        standard_grid.shift(self.grid["E2"] - standard_grid.get_origin())
        
        # Pixel Dot at (3,2) -> C5 (Calculated relative to E2)
        pixel_dot = Dot(color=COLOR_POINT)
        self.place_at_grid(pixel_dot, "C5") # Resolves Issue 35 (Move to C5)
        
        self.play(Create(standard_grid))
        self.play(Create(pixel_dot))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "We see Pixel at coordinates (3, 2)."
        self.lecture[1].set_color(COLOR_POINT)
        
        # Pixel Robot (Asset)
        # Load asset, scale 0.8, place at C5, rotate 30 deg.
        pixel_robot = SVGMobject(ROBOT_ASSET)
        self.place_at_grid(pixel_robot, "C5", scale_factor=0.8) # Resolves Issue 30 (Asset), Issue 34 (Move to C5)
        pixel_robot.rotate(30 * DEGREES)
        
        # Label (3,2) at B5 (one unit above C5)
        pixel_label = MathTex("(3, 2)", font_size=24, color=COLOR_POINT)
        self.place_at_grid(pixel_label, "B5", scale_factor=0.8) # Resolves Issue 33 (Move to B5)
        
        self.play(FadeIn(pixel_robot))
        self.play(Write(pixel_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "But from Pixel's view, his position feels different."
        self.lecture[2].set_color(COLOR_TILTED)
        
        # Tilted grid centered at same origin (E2)
        tilted_grid = NumberPlane(
            x_range=[-1, 4, 1],
            y_range=[-1, 4, 1],
            background_line_style={"stroke_color": COLOR_TILTED, "stroke_opacity": 0.3},
            axis_config={"stroke_color": COLOR_TILTED, "include_tip": True}
        )
        tilted_grid.shift(self.grid["E2"] - tilted_grid.get_origin())
        # Rotate around the origin (E2)
        tilted_grid.rotate(30 * DEGREES, about_point=self.grid["E2"])
        
        self.play(Create(tilted_grid), run_time=2)
        self.wait(2)
