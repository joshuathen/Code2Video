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
        self.setup_layout("Connecting Physics to Geometry", [
            "Mass ratio defines the wedge angle.",
            "Phase space paths form circular arcs.",
            "Arc length relates to pi digits.",
            "Collision paths mirror geometric reflections.",
            "This traces a perfect circle."
        ])
        
        # Load assets
        wedge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wedge.svg", color=WHITE)
        circle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg", color=WHITE)

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(wedge, 'D4', scale_factor=0.9)
        self.play(FadeIn(wedge))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        arc = Arc(radius=1.5, start_angle=0, angle=PI/2, color=YELLOW)
        self.place_at_grid(arc, 'C3')
        self.play(Create(arc), FadeIn(circle_icon.move_to(arc.get_center())))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        self.play(Indicate(arc))

        # === Animation for Lecture Line 4 ===
        intersection = Dot(color="#FF00FF")
        self.place_at_grid(intersection, 'E4', scale_factor=0.7)
        self.play(FadeIn(intersection))
        self.lecture[3].set_color("#FF00FF")

        # === Animation for Lecture Line 5 ===
        circle = Circle(radius=1, color=RED)
        self.place_in_area(circle, 'B3', 'E5', scale_factor=1.2)
        self.play(Create(circle))
        self.lecture[4].set_color(RED)
        self.wait(2)
