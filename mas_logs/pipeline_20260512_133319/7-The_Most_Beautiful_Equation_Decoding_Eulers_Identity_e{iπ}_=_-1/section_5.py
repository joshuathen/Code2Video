from manim import *
import numpy as np
import pathlib

# Workaround for Manim's path formatting bug
try:
    from manim import config as manim_config
    input_path_str = str(manim_config.input_file)
    if "{" in input_path_str or "}" in input_path_str:
        manim_config.input_file = pathlib.Path(input_path_str.replace("{", "_").replace("}", "_"))
except Exception:
    pass

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

class Section5Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title_text = "The Journey to π (Pi)"
        lecture_lines = [
            'We start our journey at one on the circle.',
            'Define pi as the distance halfway around.',
            'The point moves along the edge of the circle.',
            'We track its path for exactly pi units.',
            'Our journey ends at the negative one station.'
        ]
        self.setup_layout(title_text, lecture_lines)

        # Constants
        ORANGE_COLOR = "#FF8800"
        RED_COLOR = "#FF0000"
        origin = self.grid['D3']
        radius = 2.0  # Distance from D3 to D5 is 2 units in the grid
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Axes
        real_axis = Line(self.grid['D1'] + LEFT * 0.5, self.grid['D6'] + RIGHT * 0.5, color=GRAY_A)
        imag_axis = Line(self.grid['F3'] + DOWN * 0.5, self.grid['A3'] + UP * 0.5, color=GRAY_A)
        
        # Labels - satisfying Issue 41 and 42
        label_re = Text("Re", font_size=20)
        self.place_at_grid(label_re, 'D6', scale_factor=0.5)
        
        label_im = Text("Im", font_size=20)
        self.place_at_grid(label_im, 'A3', scale_factor=0.5)
        
        label_1 = Text("1", font_size=24)
        self.place_at_grid(label_1, 'D5', scale_factor=0.7)
        
        label_i = Text("i", font_size=24)
        self.place_at_grid(label_i, 'B3', scale_factor=0.7)
        
        unit_circle = Circle(radius=radius, color=BLUE_D).move_to(origin)
        
        # Grouping for Issue 40 context
        complex_plane_group = VGroup(real_axis, imag_axis, unit_circle, label_re, label_im, label_1, label_i)
        
        self.play(Create(real_axis), Create(imag_axis), FadeIn(label_re), FadeIn(label_im))
        self.play(Create(unit_circle), Write(label_1), Write(label_i))
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(ORANGE_COLOR)
        )
        # Highlight upper arc
        upper_arc = Arc(radius=radius, start_angle=0, angle=PI, color=ORANGE_COLOR, arc_center=origin)
        self.play(Create(upper_arc), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(RED_COLOR)
        )
        # Dot starting at 1 (grid D5)
        dot = Dot(self.grid['D5'], color=RED_COLOR)
        self.play(FadeIn(dot))
        # Move dot along the arc to -1 (grid D1)
        self.play(MoveAlongPath(dot, upper_arc), run_time=2)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(WHITE)
        )
        # Formula display e^{it} -> e^{iπ}
        # Unicode: \u03c0 is pi
        formula_base = MarkupText("e<sup>it</sup>", font_size=32, color=WHITE)
        self.place_at_grid(formula_base, 'A5', scale_factor=1.0)
        
        self.play(Write(formula_base))
        self.wait(0.5)
        
        formula_pi = MarkupText("e<sup>iπ</sup>", font_size=32, color=WHITE)
        self.place_at_grid(formula_pi, 'A5', scale_factor=1.0)
        
        self.play(Transform(formula_base, formula_pi))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Asset: station.svg at -1 (Issue 26)
        station = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/station.svg")
        self.place_at_grid(station, 'D1', scale_factor=0.5)
        
        label_neg_1 = Text("-1", font_size=24)
        self.place_at_grid(label_neg_1, 'D1', scale_factor=0.7)
        # Shift label slightly below the station
        label_neg_1.shift(DOWN * 0.4)
        
        # Final identity update
        formula_final = MarkupText("e<sup>iπ</sup> = -1", font_size=32, color=WHITE)
        self.place_at_grid(formula_final, 'A5', scale_factor=1.0)
        
        self.play(FadeIn(station), Write(label_neg_1))
        self.play(Transform(formula_base, formula_final))
        self.wait(2)
