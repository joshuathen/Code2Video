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
        # Initialize Scene
        title = "Practical Application: The Mechanical Torque"
        lines = [
            "Torque represents the rotational effect of a force.",
            "Maximum twist occurs when the force is perpendicular.",
            "The cross product calculates this rotational efficiency."
        ]
        self.setup_layout(title, lines)

        # Assets/Colors
        COLOR_BOLT = "#888888"
        COLOR_R = "#FF5733"
        COLOR_F = "#33FF57"
        COLOR_TAU = "#3357FF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_R)
        
        # Bolt Asset Integration
        bolt = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/bolt.svg")
        bolt.set_color(COLOR_BOLT).set_fill(COLOR_BOLT, opacity=0.8)
        self.place_at_grid(bolt, "D3", scale_factor=0.5)
        
        # Wrench Handle (spanning from bolt to D6)
        wrench_handle = Line(self.grid["D3"], self.grid["D6"], color=COLOR_BOLT, stroke_width=12)
        
        # Position Vector r
        vec_r = Arrow(self.grid["D3"], self.grid["D6"], buff=0, color=COLOR_R, stroke_width=6)
        label_r = Text("r", color=COLOR_R, font_size=24)
        self.place_at_grid(label_r, "E4", scale_factor=0.8)
        
        self.play(Create(bolt), Create(wrench_handle), Create(vec_r), Write(label_r))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_F)
        
        # Force Vector F (perpendicular to handle at D6)
        vec_f = Arrow(self.grid["D6"], self.grid["B6"], buff=0, color=COLOR_F, stroke_width=6)
        label_f = Text("F", color=COLOR_F, font_size=24)
        self.place_at_grid(label_f, "B6", scale_factor=0.8)
        
        # Perpendicular indicator (Right angle at vertex D6)
        line_r_ref = Line(self.grid["D6"], self.grid["D3"])
        right_angle = RightAngle(line_r_ref, vec_f, length=0.3, color=WHITE)
        
        self.play(Create(vec_f), Write(label_f), Create(right_angle))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_TAU)
        
        # Torque Vector Tau (Represented as out-of-page symbol at pivot column 3)
        tau_circle = Circle(radius=0.2, color=COLOR_TAU)
        tau_dot = Dot(radius=0.06, color=COLOR_TAU)
        vec_tau = VGroup(tau_circle, tau_dot)
        self.place_at_grid(vec_tau, "C3", scale_factor=1.0)
        
        # Torque formula using area-based positioning
        label_tau = Text("τ = r × F", color=COLOR_TAU, font_size=24)
        self.place_in_area(label_tau, "A2", "B4", scale_factor=0.8)
        
        # Rotation group (everything except bolt and tau indicator)
        moving_group = VGroup(wrench_handle, vec_r, vec_f, label_r, label_f, right_angle)
        
        self.play(Create(vec_tau), Write(label_tau))
        self.play(
            Rotate(moving_group, angle=30*DEGREES, about_point=self.grid["D3"]),
            run_time=2,
            rate_func=smooth
        )
        self.wait(2)
