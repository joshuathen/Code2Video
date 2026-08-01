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
        lecture_lines = [
            "Infinite sums are also called Dirichlet series.",
            "The power \"s\" determines if a series converges.",
            "The Harmonic series grows to infinity as \"s\" equals one.",
            "But the Basel series converges when \"s\" is two.",
            "Larger \"s\" values make the series shrink faster."
        ]
        self.setup_layout("Prerequisite: The Power of Series", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Dirichlet series formula
        # [Issue 25] Positioning formula to A3
        formula = Text("Σ 1/n^s", color=WHITE)
        self.place_at_grid(formula, "A3", scale_factor=1.2)
        
        self.play(
            self.lecture[0].animate.set_color(YELLOW),
            Write(formula)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Power "s" highlight
        s_part = formula[-1]
        s_circle = Circle(radius=0.2, color=BLUE).move_to(s_part)
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW),
            Create(s_circle)
        )
        self.wait(1)
        self.play(FadeOut(s_circle))

        # === Animation for Lecture Line 3 ===
        # Harmonic series s=1
        # [Issue 26] Positioning h_label to B2
        h_label = Text("s=1", color="#FF0000")
        self.place_at_grid(h_label, "B2", scale_factor=0.8)
        
        h_stack = VGroup()
        for n in range(1, 16):
            h = 1.0 / n
            rect = Rectangle(width=0.7, height=h, fill_opacity=0.8, 
                             fill_color="#FF0000", stroke_width=0.5, stroke_color=WHITE)
            if n == 1:
                rect.move_to(self.grid["F2"], aligned_edge=DOWN)
            else:
                rect.next_to(h_stack[-1], UP, buff=0)
            h_stack.add(rect)
            
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FF0000"),
            Write(h_label)
        )
        self.play(LaggedStart(*[FadeIn(r, shift=UP*0.1) for r in h_stack], lag_ratio=0.1, run_time=1.5))
        
        # [Issue 20] Scale s=1 bars to grow off-screen and overlay asset icon
        no_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/no.svg")
        self.place_at_grid(no_icon, "D2", scale_factor=0.6)
        no_icon.set_color(RED)

        self.play(
            h_stack.animate.scale(3, about_edge=DOWN).set_opacity(0.3),
            FadeIn(no_icon),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Basel series s=2
        # [Issue 27] Positioning b_label to B4
        b_label = Text("s=2", color="#00FF00")
        self.place_at_grid(b_label, "B4", scale_factor=0.8)
        
        b_stack = VGroup()
        for n in range(1, 16):
            h = 1.0 / (n**2)
            rect = Rectangle(width=0.7, height=h, fill_opacity=0.8, 
                             fill_color="#00FF00", stroke_width=0.5, stroke_color=WHITE)
            if n == 1:
                rect.move_to(self.grid["F4"], aligned_edge=DOWN)
            else:
                rect.next_to(b_stack[-1], UP, buff=0)
            b_stack.add(rect)
            
        limit_val = 1.0 * (np.pi**2 / 6)
        # Position relative to b_stack bottom
        limit_y = self.grid["F4"][1] + limit_val
        limit_line = DashedLine(
            [self.grid["F4"][0] - 0.4, limit_y, 0],
            [self.grid["F4"][0] + 0.4, limit_y, 0],
            color="#00FF00"
        )
        limit_label = Text("≈ 1.645", color="#00FF00", font_size=18).next_to(limit_line, RIGHT, buff=0.1)

        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#00FF00"),
            Write(b_label)
        )
        self.play(LaggedStart(*[FadeIn(r, shift=UP*0.1) for r in b_stack], lag_ratio=0.1, run_time=1.5))
        self.play(Create(limit_line), Write(limit_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Comparison and Conclusion
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Arrows indicating shrinking speed
        fast_indicator = Arrow(self.grid["E4"] + UP*0.5, self.grid["E4"], color="#00FF00", buff=0)
        slow_indicator = Arrow(self.grid["E2"] + UP*0.5, self.grid["E2"], color="#FF0000", buff=0)
        
        self.play(Create(fast_indicator), Create(slow_indicator))
        self.wait(1)
        self.play(FadeOut(fast_indicator), FadeOut(slow_indicator))
        
        self.wait(2)
