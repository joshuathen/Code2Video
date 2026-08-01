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
        # Setup layout with title and lecture lines
        lecture_lines = [
            "Fractals are shapes made of smaller self-similar copies.",
            "We calculate dimension using scaling and piece count.",
            "If scaling by three yields four new pieces...",
            "The dimension is log four divided by log three.",
            "This yields a fractional dimension of 1.26."
        ]
        self.setup_layout("Defining the Fractal Dimension Formula", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Line 1: Fractals are shapes made of smaller self-similar copies.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Create a straight line and transform it into a Koch segment
        base_line = Line(LEFT * 1.5, RIGHT * 1.5, color=WHITE)
        self.place_in_area(base_line, "B1", "B6")
        
        # Points for a standard Koch curve segment iteration
        kp = [
            np.array([-1.5, 0, 0]),
            np.array([-0.5, 0, 0]),
            np.array([0, 0.866, 0]),
            np.array([0.5, 0, 0]),
            np.array([1.5, 0, 0])
        ]
        koch_segment = VGroup(
            Line(kp[0], kp[1], color=WHITE),
            Line(kp[1], kp[2], color=WHITE),
            Line(kp[2], kp[3], color=WHITE),
            Line(kp[3], kp[4], color=WHITE)
        )
        self.place_in_area(koch_segment, "B1", "B6")
        
        self.play(Create(base_line))
        self.wait(0.5)
        self.play(ReplacementTransform(base_line, koch_segment))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: We calculate dimension using scaling and piece count.
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        # Using Text instead of MathTex to avoid LaTeX dependencies
        s_def = Text("s = scale factor", font_size=28, color=YELLOW)
        n_def = Text("N = number of copies", font_size=28, color=YELLOW)
        defs_group = VGroup(s_def, n_def).arrange(DOWN, aligned_edge=LEFT)
        self.place_in_area(defs_group, "D1", "D6")
        
        self.play(Write(defs_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: If scaling by three yields four new pieces...
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # Highlight N=4 and s=3
        s_val_text = Text("s = 3", color=YELLOW, font_size=32)
        n_val_text = Text("N = 4", color=YELLOW, font_size=32)
        self.place_at_grid(s_val_text, "C2")
        self.place_at_grid(n_val_text, "C5")
        
        # Highlight animation: make segments flash yellow
        self.play(Write(s_val_text), Write(n_val_text))
        for segment in koch_segment:
            self.play(segment.animate.set_color(YELLOW), run_time=0.2)
        self.play(koch_segment.animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Line 4: The dimension is log four divided by log three.
        self.play(self.lecture[3].animate.set_color(WHITE))
        
        formula = Text("D = log(N) / log(s)", color=WHITE, font_size=32)
        self.place_in_area(formula, "E1", "E6")
        
        self.play(FadeIn(formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line 5: This yields a fractional dimension of 1.26.
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        final_calc = Text("D = log(4) / log(3) ≈ 1.26", color=WHITE, font_size=32)
        self.place_in_area(final_calc, "F1", "F6")
        
        self.play(ReplacementTransform(formula, final_calc))
        self.wait(2)
