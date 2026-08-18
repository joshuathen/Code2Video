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
        lines = [
            "Rotation angle depends on light wavelength.",
            "Different colors rotate at different depths.",
            "This creates a spiraling Barber Pole effect.",
            "White light unblocks colors at different depths.",
            "Colors emerge as a vertical rainbow column."
        ]
        self.setup_layout("The Barber Pole Effect: Spectral Dependence", lines)
        
        # Define colors for lecture lines
        line_colors = ["#FF5555", "#55FF55", "#5555FF", "#FFFF55", "#FF55FF"]
        
        # Assets (Using SVG)
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")
        barberpole = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/barberpole.svg")
        lightbulb = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lightbulb.svg")
        
        # 1. Display light spectrum
        spectrum = VGroup(*[Line(UP*0.5, DOWN*0.5, color=color, stroke_width=6) for color in [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]])
        spectrum.arrange(RIGHT, buff=0.05)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(line_colors[0]))
        self.place_at_grid(prism, 'A3', scale_factor=0.3)
        self.place_at_grid(spectrum, 'B3', scale_factor=0.6)
        self.play(FadeIn(prism), Create(spectrum))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(line_colors[1]))
        arc = Arc(start_angle=0, angle=PI/4, radius=0.4, color=BLUE)
        self.place_at_grid(arc, 'B5', scale_factor=0.8)
        self.play(Create(arc))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(line_colors[2]))
        self.place_at_grid(barberpole, 'D5', scale_factor=0.4)
        self.play(FadeIn(barberpole))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(line_colors[3]))
        self.place_at_grid(lightbulb, 'E3', scale_factor=0.3)
        self.play(FadeIn(lightbulb))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(line_colors[4]))
        rainbow_col = VGroup(*[Line(UP*0.2, DOWN*0.2, color=color, stroke_width=4) for color in [RED, YELLOW, BLUE]])
        rainbow_col.arrange(DOWN, buff=0.05)
        self.place_at_grid(rainbow_col, 'D2', scale_factor=1.0)
        self.play(Create(rainbow_col))
        
        self.wait(2)
