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
        title_text = "Prerequisites: The Vector-Bot’s Movement"
        lecture_lines = [
            "Meet Vector-Bot, our guide to the linear world.",
            "He moves by scaling and adding his arrow tools.",
            "Scale a vector to change its length and direction."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # A small robot icon appears at (0,0) on a grid to represent Vector-Bot.
        # Mapping (0,0) to grid D2 (origin for this layout).
        self.lecture[0].set_color(WHITE)
        
        # Background Grid for Visuals (Belief B021: Visuals start from Column 2)
        plane = NumberPlane(
            x_range=[-1, 4, 1], 
            y_range=[-1, 2, 1],
            x_length=5, 
            y_length=3,
            axis_config={"include_numbers": False},
            background_line_style={"stroke_opacity": 0.3}
        )
        plane.shift(self.grid['D2'] - plane.get_origin())
        self.add(plane)

        # Robot construction
        robot_body = Square(side_length=0.4, color=WHITE, fill_opacity=1)
        eye1 = Dot(radius=0.04, color=BLACK).move_to(robot_body.get_center() + 0.1*UP + 0.08*LEFT)
        eye2 = Dot(radius=0.04, color=BLACK).move_to(robot_body.get_center() + 0.1*UP + 0.08*RIGHT)
        robot = VGroup(robot_body, eye1, eye2)
        
        self.place_at_grid(robot, 'D2')
        
        self.play(FadeIn(robot))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "He moves by scaling and adding his arrow tools."
        # A red arrow v from (0,0) to (1,0) stretches to (2,0) with label '2v' in #FF0000.
        # A blue arrow w appears at (0,0) to (0,1), then its tail moves to the tip of 2v at (2,0) in #0000FF.
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        
        v_color = "#FF0000"
        w_color = "#0000FF"
        
        v_start = self.grid['D2']
        v_end_1 = self.grid['D3']
        v_end_2 = self.grid['D4']
        
        v_arrow = Arrow(v_start, v_end_1, buff=0, color=v_color)
        v_label = MathTex("2\\vec{v}", color=v_color)
        # Apply Issue 20 Fix: Positioning '2v' at E2-E3
        self.place_in_area(v_label, 'E2', 'E3', scale_factor=0.8)
        
        self.play(Create(v_arrow))
        self.wait(0.5)
        # Stretch v to 2v
        self.play(
            v_arrow.animate.put_start_and_end_on(v_start, v_end_2),
            FadeIn(v_label)
        )
        self.wait(1)
        
        # Blue arrow w appears at (0,0) to (0,1)
        w_start_orig = self.grid['D2']
        w_end_orig = self.grid['C2']
        w_end_final = self.grid['C4']
        
        w_arrow = Arrow(w_start_orig, w_end_orig, buff=0, color=w_color)
        self.play(Create(w_arrow))
        self.wait(0.5)
        
        # Move tail of w to tip of 2v, and move robot to final position
        self.play(
            w_arrow.animate.put_start_and_end_on(v_end_2, w_end_final),
            robot.animate.move_to(w_end_final)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Scale a vector to change its length and direction."
        # A purple vector u appears from (0,0) to (2,1) and is labeled '2v + w' in #800080.
        # The coordinates (2,1) flash in yellow #FFFF00 next to the robot at its final position.
        self.play(self.lecture[2].animate.set_color("#800080"))
        
        u_color = "#800080"
        u_arrow = Arrow(v_start, w_end_final, buff=0, color=u_color)
        u_label = MathTex("2\\vec{v} + \\vec{w}", color=u_color)
        # Apply Issue 19 Fix: Positioning '2v + w' at B3-B4
        self.place_in_area(u_label, 'B3', 'B4', scale_factor=0.8)
        
        coords = Text("(2, 1)", color="#FFFF00", font_size=24)
        # Apply Issue 18 Fix: Positioning coordinates at D4
        self.place_at_grid(coords, 'D4', scale_factor=1.0)
        
        self.play(
            Create(u_arrow),
            FadeIn(u_label)
        )
        self.wait(0.5)
        self.play(
            Indicate(coords, color="#FFFF00"),
            FadeIn(coords)
        )
        self.wait(2)
