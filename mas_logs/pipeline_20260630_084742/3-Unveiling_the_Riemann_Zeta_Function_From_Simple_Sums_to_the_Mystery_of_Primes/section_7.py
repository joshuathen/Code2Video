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

class Section7Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "The function's zeros hide a million-dollar mathematical secret.",
            "All non-trivial zeros seem to lie on one line.",
            "This critical line has a real part of one half.",
            "Proving this would unlock the secrets of prime distribution.",
            "This remains the greatest unsolved mystery in mathematics."
        ]
        self.setup_layout("The Million Dollar Mystery: The Riemann Hypothesis", lecture_lines)

        # Pre-define elements
        # A simple coordinate plane for context
        plane = NumberPlane(
            x_range=[-1, 2, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=5,
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"include_numbers": False}
        )
        self.place_in_area(plane, "A1", "F6")

        # The Red Tightrope (Critical Line)
        # Using the midpoint of columns 3 and 4 to represent Re(s)=1/2 center
        line_top = (self.grid["A3"] + self.grid["A4"]) / 2
        line_bottom = (self.grid["F3"] + self.grid["F4"]) / 2
        tightrope = Line(line_top, line_bottom, color="#FF0000", stroke_width=4)
        
        # Zeros (Gold Coins) - Initially scattered
        scatter_positions = [
            self.grid["B2"], self.grid["C5"], self.grid["D1"], self.grid["E6"]
        ]
        zeros = VGroup(*[
            Circle(radius=0.08, color="#FFFF00", fill_opacity=1).move_to(pos)
            for pos in scatter_positions
        ])

        # Final positions on the line for alignment
        final_zeros_pos = [
            line_top + (line_bottom - line_top) * i for i in [0.2, 0.4, 0.6, 0.8]
        ]

        # Labels
        re_label = Text("Re(s) = 1/2", color=WHITE, font_size=24)
        self.place_in_area(re_label, 'F3', 'F5', scale_factor=0.7)
        
        prize_tag = Text("$1,000,000", color="#00FF00", font_size=32, weight=BOLD)
        self.place_in_area(prize_tag, 'A3', 'A5', scale_factor=0.8)

        # Glow effect
        glow_line = tightrope.copy().set_color(WHITE).set_stroke(width=12, opacity=0)

        # === Animation for Lecture Line 1 ===
        # Matching color: Yellow (#FFFF00) for coins
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.play(Create(plane), run_time=1.0)
        self.play(LaggedStart(*[FadeIn(z, scale=0.5) for z in zeros], lag_ratio=0.2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Matching color: Red (#FF0000) for tightrope
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FF0000")
        )
        self.play(Create(tightrope))
        self.play(
            *[z.animate.move_to(pos) for z, pos in zip(zeros, final_zeros_pos)],
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Matching color: White (#FFFFFF) for re_label
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFFFF")
        )
        self.play(Write(re_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Matching color: Green (#00FF00) for prize_tag
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#00FF00")
        )
        self.play(FadeIn(prize_tag, shift=UP))
        self.play(Indicate(prize_tag, color="#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Matching color: White (#FFFFFF) for glow
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color("#FFFFFF")
        )
        self.play(
            glow_line.animate.set_stroke(opacity=0.6),
            tightrope.animate.set_stroke(width=6),
            run_time=1.5
        )
        self.play(
            glow_line.animate.set_stroke(opacity=0.9),
            rate_func=there_and_back,
            run_time=2
        )
        self.wait(2)
