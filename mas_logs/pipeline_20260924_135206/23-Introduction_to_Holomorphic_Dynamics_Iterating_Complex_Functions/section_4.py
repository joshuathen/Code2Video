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
        self.setup_layout("The Julia Set: The Boundary of Chaos", [
            "The Julia set is a chaotic boundary.",
            "Iterates of points diverge at this edge.",
            "Stability defines the surrounding Fatou set."
        ])
        
        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        magnifier = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifier.svg")
        
        # Visualization objects
        points = VGroup(*[Dot(radius=0.04, color=WHITE) for _ in range(100)])
        self.place_in_area(points, "D2", "F6", scale_factor=0.5)
        
        boundary = VMobject()
        boundary.set_points_smoothly([self.grid["B3"], self.grid["B4"], self.grid["C5"], self.grid["D5"], self.grid["E4"], self.grid["E3"], self.grid["D2"], self.grid["C2"]])
        boundary.set_stroke(color=YELLOW, width=4)
        
        # === Animation for Lecture Line 1 ===
        # Use compass to orient
        self.place_at_grid(compass, "A2", scale_factor=0.5)
        self.play(FadeIn(points), FadeIn(compass))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        # Points escape to infinity (visualize as fading to dark blue)
        escaping = VGroup(*[points[i] for i in range(len(points)) if i % 3 != 0])
        self.play(escaping.animate.set_color("#00008B"))
        self.lecture[1].set_color(BLUE)

        # === Animation for Lecture Line 3 ===
        # Highlight Julia set boundary with glow and use magnifier
        self.place_at_grid(magnifier, "C4", scale_factor=0.6)
        stable = VGroup(*[points[i] for i in range(len(points)) if i % 3 == 0])
        self.play(FadeIn(boundary), FadeIn(magnifier), stable.animate.set_color(WHITE))
        self.lecture[2].set_color(WHITE)
        self.wait(1)
