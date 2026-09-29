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
        lecture_lines = ["Switch from rectangular to polar coordinates.", "The variables shift to r and θ.", "Now we see circular geometry."]
        self.setup_layout("The Polar Coordinate Transformation", lecture_lines)
        
        # Load assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        
        # Define equations
        eq1 = MathTex("x = r \\cos(\\theta), \\quad y = r \\sin(\\theta)")
        eq2 = MathTex("r, \\theta")
        eq3 = MathTex("dA = r \\, dr \\, d\\theta")
        
        # Define polar grid
        polar_grid = VGroup()
        for r in range(1, 4):
            polar_grid.add(Circle(radius=r * 0.3, color="#66FF66", stroke_width=2))
        for angle in range(0, 360, 45):
            line = Line(ORIGIN, RIGHT * 0.9, color="#66FF66", stroke_width=2)
            line.rotate(angle * DEGREES, about_point=ORIGIN)
            polar_grid.add(line)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#66FF66"))
        self.place_at_grid(compass, 'B2', scale_factor=0.5)
        self.place_at_grid(eq1, 'B3', scale_factor=0.8)
        self.play(FadeIn(compass), Write(eq1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFCC66"))
        self.place_at_grid(eq2, 'C3', scale_factor=0.8)
        self.play(FadeIn(eq2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#66CCFF"))
        self.place_at_grid(polar_grid, 'E3', scale_factor=1.0)
        self.place_at_grid(protractor, 'E5', scale_factor=0.5)
        self.place_at_grid(eq3, 'D3', scale_factor=0.8)
        self.play(Create(polar_grid), FadeIn(protractor), Write(eq3))
        self.wait(2)
