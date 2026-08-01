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

class Section5Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        self.setup_layout("The Grand Finale: e^(\u03c0i) = -1", [
            "Let's set our rotation distance to the value pi.",
            "Since pi is a half-circle, we rotate 180 degrees.",
            "We start at positive one and swing around perfectly.",
            "The journey ends exactly at the number negative one.",
            "Thus, e raised to pi-i plus one equals zero."
        ])

        # === Animation for Lecture Line 1 ===
        # Use MarkupText instead of MathTex to avoid the system 'latex' dependency
        self.play(self.lecture[0].animate.set_color("#ADD8E6"))
        formula_ix = MarkupText("e<sup>ix</sup>", color="#ADD8E6")
        self.place_in_area(formula_ix, "A3", "B4", scale_factor=1.5)
        self.play(Write(formula_ix))
        self.wait(0.5)
        
        formula_ipi_top = MarkupText("e<sup>iπ</sup>", color="#ADD8E6")
        self.place_in_area(formula_ipi_top, "A3", "B4", scale_factor=1.5)
        self.play(Transform(formula_ix, formula_ipi_top))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(WHITE))
        
        # Draw Coordinate system relative to grid
        xAxis = Line(self.grid["D1"], self.grid["D5"], color=GRAY_B)
        yAxis = Line(self.grid["F3"], self.grid["B3"], color=GRAY_B)
        unit_circle = Circle(radius=1.0, color=BLUE_E, stroke_opacity=0.3)
        self.place_at_grid(unit_circle, "D3")
        
        self.play(Create(xAxis), Create(yAxis), Create(unit_circle))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        dot = Dot(color=WHITE)
        self.place_at_grid(dot, "D4")
        
        label_1 = Text("1", font_size=24, color=WHITE)
        self.place_at_grid(label_1, "E4")
        
        self.play(FadeIn(dot), Write(label_1))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFD700"))
        
        arc = Arc(radius=1.0, start_angle=0, angle=PI, color=WHITE, arc_center=self.grid["D3"])
        
        label_i = Text("i", font_size=24, color=WHITE, slant=ITALIC)
        self.place_at_grid(label_i, "B3")

        self.play(
            MoveAlongPath(dot, arc),
            Create(arc),
            Write(label_i),
            run_time=3,
            rate_func=smooth
        )
        
        label_neg_1 = Text("-1", font_size=24, color=WHITE)
        self.place_at_grid(label_neg_1, "E2")
        self.play(Write(label_neg_1))
        
        final_eq = MarkupText("e<sup>πi</sup> = -1", color="#FFD700")
        self.place_in_area(final_eq, "F2", "F5", scale_factor=1.2)
        
        self.play(FadeOut(formula_ix), Write(final_eq))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        identity_eq = MarkupText("e<sup>πi</sup> + 1 = 0", color=WHITE)
        self.place_in_area(identity_eq, "F2", "F5", scale_factor=1.2)
        
        self.play(Transform(final_eq, identity_eq))
        self.wait(2)
