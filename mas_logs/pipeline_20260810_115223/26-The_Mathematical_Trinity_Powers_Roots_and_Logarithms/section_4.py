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
            "These are three parts of one story.",
            "Powers, roots, and logarithms unite.",
            "Base, exponent, result form a triangle.",
            "We slide the dial to transform equations.",
            "All expressions describe the same truth."
        ]
        self.setup_layout("Unified Notation: The Magic Triangle", lecture_lines)
        
        # Colors for lines
        colors = [BLUE, GREEN, YELLOW, ORANGE, RED]
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]))
        title_magic = Text("The Magic Triangle", font_size=40, color=WHITE)
        self.place_in_area(title_magic, 'A2', 'A5', scale_factor=0.9)
        self.play(FadeIn(title_magic))
        self.wait(1)
        self.play(FadeOut(title_magic))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]))
        # Use an SVG placeholder if standard polygon was requested via 'triangle_animation' variable
        triangle_animation = Polygon(
            self.grid['B4'], self.grid['E2'], self.grid['E6'],
            color=WHITE
        )
        self.place_in_area(triangle_animation, 'B3', 'F5', scale_factor=0.8)
        
        label_a = Text("Base", font_size=24, color=WHITE).next_to(triangle_animation.get_top(), UP)
        label_b = Text("Exponent", font_size=24, color=WHITE).next_to(triangle_animation.get_left(), LEFT)
        label_c = Text("Result", font_size=24, color=WHITE).next_to(triangle_animation.get_right(), RIGHT)
        
        self.play(Create(triangle_animation), Write(label_a), Write(label_b), Write(label_c))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]))
        self.play(triangle_animation.animate.set_color(YELLOW))
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(colors[3]))
        dial = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dial.svg")
        self.place_at_grid(dial, 'D3', scale_factor=0.7)
        self.play(FadeIn(dial))
        self.play(dial.animate.shift(RIGHT * 0.5), run_time=0.5)
        self.play(dial.animate.shift(LEFT * 0.5), run_time=0.5)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(colors[4]))
        self.play(Flash(triangle_animation, color=RED, num_lines=10))
        self.wait(2)
