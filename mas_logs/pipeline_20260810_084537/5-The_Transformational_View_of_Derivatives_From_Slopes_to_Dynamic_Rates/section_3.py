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
            "The derivative is a dynamic transformation operator.",
            "It maps position to its rate of change.",
            "Like a chameleon's tongue, capturing instantaneous speed.",
            "It transforms functions into their velocity graphs.",
            "A micro-second map of the function's trend."
        ]
        self.setup_layout("Defining the Derivative as a Dynamic Operator", lecture_lines)
        
        # Elements
        chameleon_img = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/chameleon.png")
        tongue_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tongue.svg")
        
        derivative_text = Text("Instantaneous Rate of Change", color=BLUE)
        limit_def = MathTex(
            r"f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}", 
            color=WHITE
        )
        
        # VideoCritic requested elements
        highlight_circle = Circle(color=PURPLE, radius=0.5)
        # Using placeholder for grid_group as it wasn't explicitly provided
        grid_group = VGroup(Rectangle(width=2, height=2, color=GRAY), Text("Grid")) 

        # === Animation for Lecture Line 1 ===
        # The derivative is a dynamic transformation operator.
        self.play(FadeIn(self.place_in_area(derivative_text, 'B2', 'B5', 0.8)),
                  FadeIn(self.place_at_grid(chameleon_img, 'A1', 0.5)))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # It maps position to its rate of change.
        # Fixed per Issue 27/38
        self.play(FadeIn(self.place_in_area(limit_def, 'D2', 'F5', 0.9)))
        self.lecture[1].set_color(WHITE)

        # === Animation for Lecture Line 3 ===
        # Like a chameleon's tongue, capturing instantaneous speed.
        # Fixed per Issue 28/38
        self.play(FadeIn(self.place_at_grid(highlight_circle, 'E3', 0.5)))
        self.lecture[2].set_color(GREEN)

        # === Animation for Lecture Line 4 ===
        # It transforms functions into their velocity graphs.
        # Fixed per Issue 29/38
        self.play(FadeIn(self.place_in_area(grid_group, 'A1', 'F6', 0.75)))
        self.lecture[3].set_color("#FF00FF")

        # === Animation for Lecture Line 5 ===
        # A micro-second map of the function's trend.
        self.play(Flash(tongue_svg.move_to(self.grid['C3']), color=WHITE, line_length=0.2, flash_radius=0.5))
        self.lecture[4].set_color(YELLOW)
        
        self.wait(2)
