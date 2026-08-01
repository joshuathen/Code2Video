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
        # Initial Setup
        title = "The Quest of the Delivery Drone"
        lines = [
            "A drone needs to reach target point b.",
            "Two thrusters move in fixed directions, v1 and v2.",
            "We must find the power x and y required."
        ]
        self.setup_layout(title, lines)
        
        # Define Colors
        color_v1 = "#00FF00"
        color_v2 = "#0000FF"
        color_target = RED
        color_eq = WHITE
        
        # Initialize lecture colors (dimmed)
        for line in self.lecture:
            line.set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        # Fade in a drone icon at the origin and a target marker at (7, 6).
        
        # Create Coordinate System
        plane = NumberPlane(
            x_range=[0, 8, 1],
            y_range=[0, 8, 1],
            x_length=4.5,
            y_length=4.5,
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"include_tip": True}
        )
        # Fix for Issue 24: Adjust scale to avoid tight boundaries
        self.place_in_area(plane, 'B1', 'F6', scale_factor=0.8)
        
        # Drone at origin
        drone = Triangle(color=WHITE, fill_opacity=1).scale(0.1)
        drone.move_to(plane.c2p(0, 0))
        drone_label = Text("Drone", font_size=16, color=WHITE).next_to(drone, DOWN, buff=0.1)
        
        # Target marker at (7,6)
        target_marker = Cross(stroke_width=3, color=color_target).scale(0.15)
        target_marker.move_to(plane.c2p(7, 6))
        target_label = MathTex(r"\vec{b} = \begin{bmatrix} 7 \\ 6 \end{bmatrix}", font_size=20, color=WHITE).next_to(target_marker, RIGHT, buff=0.1)

        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(
            Create(plane),
            FadeIn(drone),
            Write(drone_label),
            FadeIn(target_marker),
            Write(target_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Draw vector v1=[2,1] in #00FF00 and vector v2=[1,2] in #0000FF from the origin.
        
        v1_vec = Arrow(plane.c2p(0,0), plane.c2p(2,1), buff=0, color=color_v1, stroke_width=4)
        v2_vec = Arrow(plane.c2p(0,0), plane.c2p(1,2), buff=0, color=color_v2, stroke_width=4)
        
        v1_label = MathTex(r"\vec{v}_1", font_size=20, color=color_v1).next_to(v1_vec.get_end(), RIGHT, buff=0.1)
        v2_label = MathTex(r"\vec{v}_2", font_size=20, color=color_v2).next_to(v2_vec.get_end(), UP, buff=0.1)

        self.play(self.lecture[1].animate.set_color(WHITE))
        self.play(
            GrowArrow(v1_vec),
            Write(v1_label),
            run_time=1
        )
        self.play(
            GrowArrow(v2_vec),
            Write(v2_label),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display the equation x[2, 1] + y[1, 2] = [7, 6] in #FFFFFF at the top.
        
        equation = MathTex(
            r"x \begin{bmatrix} 2 \\ 1 \end{bmatrix} + y \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 7 \\ 6 \end{bmatrix}",
            font_size=28, color=color_eq
        )
        # Fix for Issue 23: Scale down to avoid crowding the title
        self.place_in_area(equation, 'A1', 'A6', scale_factor=0.7)
        
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.play(Write(equation))
        self.wait(2)
