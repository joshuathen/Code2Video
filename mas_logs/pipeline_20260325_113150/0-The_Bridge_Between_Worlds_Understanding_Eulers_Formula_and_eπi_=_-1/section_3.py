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
        # Setup layout with the required lecture lines
        lecture_lines = [
            "Real exponential growth pushes a point forward linearly.",
            "Imaginary growth pushes the point sideways at right angles.",
            "This continuous sideways push results in perfect circular motion."
        ]
        self.setup_layout("The Intuition of 'e' and Growth", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Real exponential growth pushes a point forward linearly.
        self.lecture[0].set_color("#FF00FF")
        formula_ex = Text("e^x", color="#FF00FF")
        self.place_at_grid(formula_ex, "A4", scale_factor=1.2)
        
        # Create a horizontal line path representing linear growth
        line_path = Line(self.grid["C2"], self.grid["C5"], color="#FF00FF")
        dot = Dot(color="#FF00FF")
        self.place_at_grid(dot, "C2")
        
        self.play(Write(formula_ex), Create(line_path), FadeIn(dot))
        self.play(dot.animate.move_to(self.grid["C5"]), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Imaginary growth pushes the point sideways at right angles.
        self.lecture[1].set_color("#00FFFF")
        formula_eit = Text("e^it", color="#00FFFF")
        self.place_at_grid(formula_eit, "A4", scale_factor=1.2)
        
        # Center of the circular growth at grid D3
        center_pos = self.grid["D3"]
        # Initial position of the dot for rotation at grid D5
        dot_start_pos = self.grid["D5"]
        
        dot_new = Dot(color="#00FFFF")
        self.place_at_grid(dot_new, "D5")
        
        # Vectors to demonstrate the perpendicular push
        pos_vec = Arrow(center_pos, dot_start_pos, buff=0, color=GRAY, stroke_width=2)
        # Velocity vector points from D5 to B5 (perpendicular to radial vector pos_vec)
        vel_vec = Arrow(dot_start_pos, self.grid["B5"], buff=0, color="#00FFFF", stroke_width=4)
        vel_label = Text("i", color="#00FFFF")
        # Position label near the velocity vector (C5 is between B5 and D5)
        self.place_at_grid(vel_label, "C5", scale_factor=0.8)

        self.play(
            Transform(formula_ex, formula_eit),
            FadeOut(line_path),
            FadeOut(dot),
            FadeIn(dot_new),
            FadeIn(pos_vec),
            FadeIn(vel_vec),
            FadeIn(vel_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This continuous sideways push results in perfect circular motion.
        self.lecture[2].set_color("#00FFFF")
        
        # Group elements to rotate them together around the center
        moving_group = VGroup(dot_new, pos_vec, vel_vec, vel_label)
        # Trace the path to visualize the circular orbit
        path = TracedPath(dot_new.get_center, stroke_color="#00FFFF", stroke_width=3)
        self.add(path)
        
        # Complete the rotation to demonstrate the circular orbit
        self.play(Rotate(moving_group, angle=2*PI, about_point=center_pos), run_time=4, rate_func=linear)
        self.wait(2)
