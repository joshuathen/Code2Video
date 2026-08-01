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
        self.setup_layout(
            "The Perspective Problem: Maya and the Drone", 
            [
                "Maya and her drone see the world differently.", 
                "The drone is tilted, viewing targets from its perspective.", 
                "We need a translator between these two viewpoints."
            ]
        )
        
        # Grid center for consistent placement (calculated as center of A1 to F6)
        grid_center = np.array([3.0, -0.3, 0])
        
        # Colors
        MAYA_COLOR = "#87CEEB"
        DRONE_COLOR = "#FFD700"
        VECTOR_COLOR = "#FF0000"
        TEXT_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(MAYA_COLOR)
        
        maya_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": MAYA_COLOR, "stroke_opacity": 0.4},
            axis_config={"stroke_color": MAYA_COLOR}
        )
        self.place_in_area(maya_grid, 'A1', 'F6', scale_factor=1.0)
        
        maya_label = Text("Maya's View", font_size=24, color=TEXT_COLOR)
        self.place_in_area(maya_label, 'A1', 'A3', scale_factor=0.7)
        
        maya_dot = Dot(point=grid_center, color=MAYA_COLOR)
        maya_dot_label = Text("Maya", font_size=16, color=MAYA_COLOR).next_to(maya_dot, DOWN, buff=0.1)
        
        self.play(
            Create(maya_grid),
            Write(maya_label),
            FadeIn(maya_dot),
            Write(maya_dot_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(DRONE_COLOR)
        
        drone_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": DRONE_COLOR, "stroke_opacity": 0.4},
            axis_config={"stroke_color": DRONE_COLOR}
        ).rotate(PI/4).move_to(grid_center)
        
        drone_label = Text("Drone's View", font_size=24, color=TEXT_COLOR)
        self.place_in_area(drone_label, 'A4', 'A6', scale_factor=0.7)
        
        self.play(
            Create(drone_grid),
            Write(drone_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(VECTOR_COLOR)
        
        # The vector is 1 unit along the drone's forward axis (rotated 45 degrees)
        # Vector in standard basis: [cos(45), sin(45)]
        target_vec = Arrow(
            start=grid_center,
            end=grid_center + np.array([np.cos(PI/4), np.sin(PI/4), 0]),
            buff=0,
            color=VECTOR_COLOR,
            stroke_width=6
        )
        
        # Display Coordinates
        drone_coords = Text("Drone: [1, 0]", color=DRONE_COLOR, font_size=24)
        maya_coords = Text("Maya: [0.7, 0.7]", color=MAYA_COLOR, font_size=24)
        
        self.place_in_area(drone_coords, 'F1', 'F3', scale_factor=0.8)
        self.place_in_area(maya_coords, 'F4', 'F6', scale_factor=0.8)
        
        self.play(
            GrowArrow(target_vec),
            Write(drone_coords),
            Write(maya_coords)
        )
        self.wait(2)
