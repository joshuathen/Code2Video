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
        lecture_lines = [
            "Large mass ratios reveal digits of pi.",
            "One hundred ratio gives thirty-one collisions.",
            "Ten thousand ratio gives three hundred fourteen.",
            "The geometry matches the arctangent expansion.",
            "Digits of pi appear in the collisions."
        ]
        self.setup_layout("The Revelation: Connecting to Pi", lecture_lines)
        
        # Load Assets
        pendulum_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pendulum.svg")
        scale_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg")
        
        # Elements
        blue_circle = Ellipse(width=3, height=2, color=BLUE)
        yellow_dots = VGroup(Dot(color=YELLOW), Dot(color=WHITE)).arrange(RIGHT)
        circle_arc = Arc(radius=1.5, start_angle=0, angle=PI/2, color=YELLOW)
        pi_text = MathTex(r"\\pi \\approx 3.14159", color="#FF00FF")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_in_area(blue_circle, 'A4', 'C6', scale_factor=0.9) # Adjusting to grid 4-6 range
        self.play(Create(blue_circle), FadeIn(pendulum_icon.next_to(blue_circle, UP)))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.place_at_grid(yellow_dots, 'B5', scale_factor=0.6)
        self.play(FadeIn(yellow_dots))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        self.place_in_area(circle_arc, 'D4', 'F6', scale_factor=0.6)
        self.play(Create(circle_arc))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(ORANGE)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF00FF")
        self.place_at_grid(pi_text, 'D3', scale_factor=1.0)
        self.play(Write(pi_text), FadeIn(scale_icon.next_to(pi_text, DOWN)))
        self.wait(2)
