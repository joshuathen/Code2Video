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
        self.setup_layout(
            "Establishing the 'Obvious' Pattern", 
            [
                "With one point, we have just one region.",
                "Two points create a chord and two regions.",
                "Three points divide the circle into four distinct areas.",
                "Four points yield eight regions, and five yield sixteen.",
                "The pattern seems clear: the regions double every time."
            ]
        )

        # Define colors
        HIGHLIGHT_COLOR = "#0000FF"  # Blue as requested
        TEXT_COLOR = WHITE

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        # Table Header
        n_header = MathTex("n", color=WHITE)
        r_header = MathTex("R", color=WHITE)
        self.place_at_grid(n_header, "A4")
        self.place_at_grid(r_header, "A5") # Issue 25: Move R column to 5
        
        # Adjusting line to fit the new table layout
        header_line = Line(
            self.grid["A4"] + LEFT * 0.3, 
            self.grid["A5"] + RIGHT * 0.3, 
            color=WHITE
        ).shift(DOWN * 0.4)
        
        # n=1 Row
        n1 = MathTex("1", color=WHITE)
        r1 = MathTex("1", color=WHITE)
        self.place_at_grid(n1, "B4")
        self.place_at_grid(r1, "B5") # Issue 25: Move R column to 5
        
        # Circle Diagram for n=1
        c1 = Circle(radius=0.8, color=WHITE)
        p1 = Dot(c1.point_at_angle(90 * DEGREES), color=WHITE)
        diag1 = VGroup(c1, p1)
        self.place_in_area(diag1, "A1", "C3", scale_factor=0.8) # Issue 23: Provide more breathing room

        self.play(
            Create(n_header), 
            Create(r_header), 
            Create(header_line),
            Write(n1), 
            Write(r1),
            FadeIn(diag1)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # n=2 row
        n2 = MathTex("2", color=WHITE)
        r2 = MathTex("2", color=WHITE)
        self.place_at_grid(n2, "C4")
        self.place_at_grid(r2, "C5") # Issue 25: Move R column to 5
        
        # Circle Diagram for n=2
        c2 = Circle(radius=0.8, color=WHITE)
        p2_1 = Dot(c2.point_at_angle(90 * DEGREES), color=WHITE)
        p2_2 = Dot(c2.point_at_angle(270 * DEGREES), color=WHITE)
        chord2 = Line(p2_1.get_center(), p2_2.get_center(), color=WHITE)
        diag2 = VGroup(c2, p2_1, p2_2, chord2)
        self.place_in_area(diag2, "A1", "C3", scale_factor=0.8) # Issue 23

        self.play(FadeOut(diag1), FadeIn(diag2), Write(n2), Write(r2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # n=3 row
        n3 = MathTex("3", color=WHITE)
        r3 = MathTex("4", color=WHITE)
        self.place_at_grid(n3, "D4")
        self.place_at_grid(r3, "D5") # Issue 25: Move R column to 5

        # Circle Diagram for n=3
        c3 = Circle(radius=0.8, color=WHITE)
        p3_angles = [90, 210, 330]
        p3_dots = [Dot(c3.point_at_angle(a * DEGREES), color=WHITE) for a in p3_angles]
        chord3_1 = Line(p3_dots[0].get_center(), p3_dots[1].get_center(), color=WHITE)
        chord3_2 = Line(p3_dots[1].get_center(), p3_dots[2].get_center(), color=WHITE)
        chord3_3 = Line(p3_dots[2].get_center(), p3_dots[0].get_center(), color=WHITE)
        diag3 = VGroup(c3, *p3_dots, chord3_1, chord3_2, chord3_3)
        self.place_in_area(diag3, "A1", "C3", scale_factor=0.8) # Issue 23

        self.play(FadeOut(diag2), FadeIn(diag3), Write(n3), Write(r3))
        self.wait(1)
        self.play(FadeOut(diag3))

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        n4 = MathTex("4", color=WHITE)
        r4 = MathTex("8", color=WHITE)
        n5 = MathTex("5", color=WHITE)
        r5 = MathTex("16", color=WHITE)
        self.place_at_grid(n4, "E4")
        self.place_at_grid(r4, "E5") # Issue 25
        self.place_at_grid(n5, "F4")
        self.place_at_grid(r5, "F5") # Issue 25

        self.play(Write(n4), Write(r4))
        self.wait(0.5)
        self.play(Write(n5), Write(r5))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(HIGHLIGHT_COLOR)
        )

        # Draw blue box around R values
        r_values = VGroup(r1, r2, r3, r4, r5)
        box = SurroundingRectangle(r_values, color=HIGHLIGHT_COLOR, buff=0.1)
        
        # Predicted formula
        formula = MathTex("R = 2^{n-1}?", color=HIGHLIGHT_COLOR)
        # Issue 24: Move the formula to the lower-left grid area
        self.place_in_area(formula, "D1", "F3", scale_factor=1.0)

        self.play(Create(box))
        self.play(Write(formula))
        self.wait(2)

        # Cleanup highlights
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
