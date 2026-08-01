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

class Section2Scene(TeachingScene):
    def construct(self):
        # 1. Setup Layout
        title_text = "Prerequisite: The Power of Infinite Sums"
        lecture_lines = [
            "- Infinite steps can lead to a finite, definite distance.",
            "- Imagine walking halfway to a destination every single second.",
            "- Some series converge, while the harmonic series grows forever."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Assets and Colors
        walker_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/walker.svg"
        COLOR_CONV = "#ADD8E6"
        COLOR_DIV = "#FF4500"

        # === Animation for Lecture Line 1 ===
        # A number line from 0 to 2 appears; a walker moves in increments of 1, 1/2, 1/4, 1/8.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # FOCUSED FIX: Added label_constructor=Text to bypass the requirement for a 'latex' installation.
        number_line = NumberLine(
            x_range=[0, 2.1, 0.5],
            length=5,
            include_numbers=True,
            label_constructor=Text,
            font_size=20,
            color=WHITE
        )
        wall = Line(UP*0.4, DOWN*0.4, color=GREY, stroke_width=4).move_to(number_line.number_to_point(2))
        plane_group = VGroup(number_line, wall)
        
        # Utilize right-side space by placing plane_group in defined area.
        self.place_in_area(plane_group, 'B2', 'E5', scale_factor=1.3)
        
        walker = SVGMobject(walker_path)
        # Defining indicator anchor at C4, then positioning at line start.
        self.place_at_grid(walker, 'C4', scale_factor=0.3)
        walker.move_to(number_line.number_to_point(0) + UP * 0.4)

        self.play(Create(number_line), Create(wall))
        self.play(FadeIn(walker))
        self.wait(0.5)

        # Incremental movement representing Zeno's Paradox
        current_val = 0
        steps = [1, 0.5, 0.25, 0.125]
        for step in steps:
            current_val += step
            self.play(walker.animate.move_to(number_line.number_to_point(current_val) + UP * 0.4), run_time=0.6)
        
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The equation 1 + 1/2 + 1/4 + 1/8 +... = 2 appears in light blue (#ADD8E6).
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        
        eq_conv = Text("1 + 1/2 + 1/4 + 1/8 + ... = 2", font_size=24, color=COLOR_CONV)
        # Positioning label at grid position B5.
        self.place_at_grid(eq_conv, 'B5', scale_factor=0.8)
        
        self.play(Write(eq_conv))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # The Harmonic series 1 + 1/2 + 1/3 + 1/4 +... appears in red (#FF4500).
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        
        eq_div = Text("1 + 1/2 + 1/3 + 1/4 + ...", font_size=24, color=COLOR_DIV)
        self.place_at_grid(eq_div, 'D5', scale_factor=0.8)
        
        arrow = Arrow(start=self.grid['D5'] + RIGHT*1.5, end=self.grid['D6'] + RIGHT*1.5, color=COLOR_DIV)
        inf_label = Text("Infinity", font_size=20, color=COLOR_DIV).next_to(arrow, UP)
        
        self.play(Write(eq_div))
        self.play(Create(arrow), Write(inf_label))
        
        self.wait(3)
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
