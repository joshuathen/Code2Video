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
        self.setup_layout("The Paradox of Dimensionality", ["A 1D line is just length.", "A 2D plane covers area.", "Can 1D lines fill 2D squares?"])
        
        # Elements using assets
        # Path references from storyboard:
        # Paper: /scratch/pawsey1357/jthen/Code2Video/assets/icon/paper.svg
        # Ruler: /scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg
        
        # Using placeholder shapes if SVGs are not immediately available as mobjects, 
        # but the prompt requires using Asset references.
        # Since I am in a restricted environment, I will attempt to load them.
        try:
            square = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/paper.svg", color=WHITE)
            line = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg", color=YELLOW)
        except:
            square = Square(side_length=1.5, color=WHITE, stroke_width=2)
            line = Line(start=LEFT*0.75, end=RIGHT*0.75, color=YELLOW, stroke_width=4)

        # Apply positioning constraints
        self.place_in_area(square, 'B2', 'D4', scale_factor=0.5)
        square_label = Text("Space", font_size=24, color=WHITE).next_to(square, UP)
        
        self.place_at_grid(line, 'F2', scale_factor=0.6)
        line_label = Text("Data", font_size=24, color=YELLOW)
        self.place_at_grid(line_label, 'F5', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(square), Write(square_label), FadeIn(line), Write(line_label))
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        self.play(square.animate.set_color(BLUE), square_label.animate.set_color(BLUE))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        
        # Simulation of line filling space
        path = VMobject()
        path.set_points_as_corners([
            square.get_corner(DL) + UP*0.1 + RIGHT*0.1,
            square.get_corner(DR) + UP*0.1 + LEFT*0.1,
            square.get_corner(DR) + UP*0.4 + LEFT*0.1,
            square.get_corner(DL) + UP*0.4 + RIGHT*0.1,
            square.get_corner(DL) + UP*0.7 + RIGHT*0.1,
            square.get_corner(DR) + UP*0.7 + LEFT*0.1,
        ])
        path.set_stroke(YELLOW, width=3)
        
        self.play(ReplacementTransform(line, path), line_label.animate.set_color(RED))
        self.play(Create(path), run_time=2)
        self.wait(2)
