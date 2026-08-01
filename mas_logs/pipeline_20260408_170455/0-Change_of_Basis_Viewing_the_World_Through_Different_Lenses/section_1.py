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
        lecture_lines = [
            'One location can be described by different coordinate systems.',
            'The drone follows a grid aligned with the streets.',
            'The bird sees the world through a tilted grid.',
            'Both observers point to the same physical location.',
            'Same spot, but different coordinates based on their perspective.'
        ]
        self.setup_layout("The Concept of 'Perspective'", lecture_lines)
        
        # Colors
        GRAY_COLOR = "#555555"
        BLUE_COLOR = "#0000FF"
        YELLOW_COLOR = "#FFFF00"
        
        # Grid area reference
        grid_area_tl = "A1"
        grid_area_br = "F6"

        # === Animation for Lecture Line 1 ===
        drone_grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={"stroke_color": GRAY_COLOR, "stroke_width": 2},
            axis_config={"include_numbers": False, "stroke_color": GRAY_COLOR}
        ).scale(0.6)
        self.place_in_area(drone_grid, grid_area_tl, grid_area_br)
        
        drone_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/drone.svg")
        drone_asset.scale(0.2)
        # Position the drone at (3, 2) relative to the drone_grid
        drone_asset.move_to(drone_grid.c2p(3, 2))
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            FadeIn(drone_grid),
            FadeIn(drone_asset)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        drone_arrow = Arrow(
            start=drone_grid.get_origin(),
            end=drone_asset.get_center(),
            buff=0,
            color=BLUE_COLOR,
            stroke_width=4
        )
        drone_label = Text("Drone: [3, 2]", font_size=20, color=BLUE_COLOR)
        # Fix for Issue 27: Move label from B6 to B5 and scale
        self.place_at_grid(drone_label, "B5", scale_factor=0.8)
        
        self.play(
            self.lecture[0].animate.set_color(GRAY_COLOR),
            self.lecture[1].animate.set_color(BLUE_COLOR),
            Create(drone_arrow),
            FadeIn(drone_label)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        bird_grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={"stroke_color": YELLOW_COLOR, "stroke_width": 1, "stroke_opacity": 0.4},
            axis_config={"include_numbers": False, "stroke_color": YELLOW_COLOR}
        ).scale(0.6).rotate(30 * DEGREES)
        self.place_in_area(bird_grid, grid_area_tl, grid_area_br)
        
        bird_origin_marker = Triangle(color=YELLOW_COLOR, fill_opacity=0.8).scale(0.12).move_to(drone_grid.get_origin())
        
        self.play(
            self.lecture[1].animate.set_color(GRAY_COLOR),
            self.lecture[2].animate.set_color(YELLOW_COLOR),
            FadeIn(bird_grid),
            FadeIn(bird_origin_marker)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 4 ===
        # Bird coordinates calculation: (3, 2) rotated by -30 degrees
        # x' = 3*cos(30) + 2*sin(30) = 3*0.866 + 2*0.5 = 2.598 + 1 = 3.598
        # y' = -3*sin(30) + 2*cos(30) = -3*0.5 + 2*0.866 = -1.5 + 1.732 = 0.232
        bird_arrow = Arrow(
            start=bird_grid.get_origin(),
            end=drone_asset.get_center(),
            buff=0,
            color=YELLOW_COLOR,
            stroke_width=4
        )
        bird_label = Text("Bird: [3.60, 0.23]", font_size=20, color=YELLOW_COLOR)
        # Fix for Issue 28: Move label from C6 to C5 and scale
        self.place_at_grid(bird_label, "C5", scale_factor=0.8)
        
        self.play(
            self.lecture[2].animate.set_color(GRAY_COLOR),
            self.lecture[3].animate.set_color(WHITE),
            Create(bird_arrow),
            FadeIn(bird_label)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(GRAY_COLOR),
            self.lecture[4].animate.set_color(WHITE)
        )
        
        # Flash the drone repeatedly
        for _ in range(2):
            self.play(Flash(drone_asset, color=WHITE, flash_radius=0.4), run_time=0.5)
            self.wait(0.2)
            
        self.wait(2)
