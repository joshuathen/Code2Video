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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Geometric Interpretation of Pi", [
            "Large mass ratios create a full semicircle path.",
            "Total collisions reveal digits of Pi.",
            "Geometric arc length maps perfectly to Pi."
        ])
        
        # Colors
        color_1 = "#40E0D0" # Turquoise
        color_2 = "#FF7F50" # Coral
        color_3 = "#FF0000" # Red
        
        # --- Visual Setup ---
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg]
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg", color=WHITE)
        # Apply fix for Issue 30 and 32 (Grid space utilization)
        self.place_in_area(circle, "B3", "E6", scale_factor=0.6)
        
        # Trace path (semicircle)
        arc = Arc(radius=1.5, start_angle=PI, angle=PI, color=color_1)
        # Ensure arc matches the circle's center
        arc.move_to(circle.get_center())
        
        # --- Animation for Lecture Line 1 ---
        self.lecture[0].set_color(color_1)
        self.play(Create(circle), Create(arc), run_time=2)
        
        # --- Animation for Lecture Line 2 ---
        self.lecture[1].set_color(color_2)
        # Marking collisions as points on the circle
        collision_points = VGroup(*[
            Dot(color=color_2, radius=0.08).move_to(arc.point_from_proportion(i/10))
            for i in range(11)
        ])
        self.play(FadeIn(collision_points), run_time=2)
        
        # --- Animation for Lecture Line 3 ---
        self.lecture[2].set_color(color_3)
        pi_text = MathTex(r"\\pi", color=color_3, font_size=72)
        # Apply fix for Issue 31
        self.place_at_grid(pi_text, "E5", scale_factor=0.7)
        self.play(Write(pi_text), run_time=2)
        self.wait(2)
