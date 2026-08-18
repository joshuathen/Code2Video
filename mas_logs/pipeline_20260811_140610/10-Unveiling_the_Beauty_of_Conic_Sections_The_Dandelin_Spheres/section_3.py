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
            "Tangency points on the plane become the foci.",
            "Distance sum to foci is constant for ellipses.",
            "This length equals the distance between circles of tangency."
        ]
        self.setup_layout("Connecting the Dots: The Ellipse Proof", lecture_lines)
        
        # Ellipse
        ellipse = Ellipse(width=3, height=2, color=WHITE)
        self.place_in_area(ellipse, "B3", "E6", scale_factor=0.7)
        
        # Foci as SVG assets
        f1_sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        f2_sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        
        self.place_at_grid(f1_sphere, "C2", scale_factor=0.2)
        self.place_at_grid(f2_sphere, "C6", scale_factor=0.2)
        
        f1_label = Text("F1", font_size=16, color=YELLOW).next_to(f1_sphere, UP, buff=0.1)
        f2_label = Text("F2", font_size=16, color=YELLOW).next_to(f2_sphere, UP, buff=0.1)
        
        # Point on ellipse
        p = Dot(color=WHITE)
        p.move_to(ellipse.point_from_proportion(0.2))
        
        # Connecting lines
        line1 = Line(p.get_center(), f1_sphere.get_center(), color="#33A1FF")
        line2 = Line(p.get_center(), f2_sphere.get_center(), color="#33A1FF")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(ellipse))
        self.play(FadeIn(f1_sphere, f1_label, f2_sphere, f2_label))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#33A1FF"))
        self.play(Create(p))
        self.play(Create(line1), Create(line2))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        # Animate point moving
        self.play(
            MoveAlongPath(p, ellipse),
            UpdateFromFunc(line1, lambda l: l.put_start_and_end_on(p.get_center(), f1_sphere.get_center())),
            UpdateFromFunc(line2, lambda l: l.put_start_and_end_on(p.get_center(), f2_sphere.get_center())),
            run_time=4
        )
        self.wait(1)
