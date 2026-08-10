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
        lecture_lines = [
            "The mass ratio determines the wedge angle.",
            "Increasing mass makes the wedge very narrow.",
            "Reflections bounce back and forth inside.",
            "The number of bounces measures the arc.",
            "It behaves like a bouncing billiard ball."
        ]
        self.setup_layout("The Geometric Connection", lecture_lines)
        
        # Asset Loading
        billiard_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/billiard.svg")
        ball_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF00FF")
        wedge = Sector(start_angle=0, angle=PI/6, color="#FF00FF", fill_opacity=0.3)
        self.place_in_area(wedge, "B3", "D5", scale_factor=0.9)
        self.place_at_grid(billiard_icon, "B3", scale_factor=0.5)
        self.play(Create(wedge), FadeIn(billiard_icon))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        new_wedge = Sector(start_angle=0, angle=PI/12, color="#FF00FF", fill_opacity=0.3)
        self.place_in_area(new_wedge, "B3", "D5", scale_factor=0.9)
        self.play(Transform(wedge, new_wedge))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        self.place_at_grid(ball_icon, "C4", scale_factor=0.4)
        self.play(FadeIn(ball_icon))
        self.play(ball_icon.animate.move_to(self.grid["C5"]), run_time=0.5)
        self.play(ball_icon.animate.move_to(self.grid["D4"]), run_time=0.5)
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        label = Text("Arc Length", font_size=24, color="#00FF00")
        self.place_at_grid(label, "A5", scale_factor=0.6)
        self.play(Write(label))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF9900")
        # Grid reference labels mentioned in issue 29
        grid_ref = Text("Grid Visual (A1-F6)", font_size=18, color=GRAY)
        self.place_in_area(grid_ref, "E1", "F6", scale_factor=0.5)
        self.play(FadeIn(grid_ref))
