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
            "Collision paths form an arc.",
            "Energy creates an elliptical path.",
            "Momentum forms boundary lines.",
            "Path crossings count each collision.",
            "The geometry encodes the total count."
        ]
        self.setup_layout("Geometry of Collisions", lecture_lines)
        
        # Prepare geometric elements
        ellipse = Ellipse(width=3, height=2, color=BLUE)
        boundary_line = Line(start=np.array([-2, -1, 0]), end=np.array([2, 1, 0]), color=YELLOW)
        arc = Arc(radius=1, start_angle=0, angle=PI/2, color=GREEN)
        particle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg", color=WHITE)
        dot = Dot(color="#FF0000")
        
        # Placement following fixes
        self.place_in_area(ellipse, 'B3', 'D5', scale_factor=0.9)
        self.place_at_grid(boundary_line, 'C4', scale_factor=1.0)
        self.place_at_grid(arc, 'C3', scale_factor=0.8)
        self.place_at_grid(particle, 'B2', scale_factor=0.3)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(GREEN)
        self.play(Create(arc), FadeIn(particle))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        self.play(Create(ellipse))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.play(Create(boundary_line))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF0000")
        self.place_at_grid(dot, 'C3', scale_factor=0.5)
        self.play(FadeIn(dot))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        self.play(Indicate(self.lecture[4]))
        self.wait(1)
