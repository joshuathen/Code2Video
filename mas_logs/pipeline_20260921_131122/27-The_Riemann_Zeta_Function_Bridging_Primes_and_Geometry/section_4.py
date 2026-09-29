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
        self.setup_layout("The Riemann Hypothesis: The $1 Million Mystery", [
            "The critical strip holds unknown zeros.", 
            "Riemann conjectured zeros lie on one line.", 
            "This line is at Re(s) = 1/2."
        ])
        
        # --- Visual Setup ---
        # The critical strip
        strip = Rectangle(width=0.8, height=4.5, color="#00FF00", fill_opacity=0.2)
        self.place_in_area(strip, "C2", "F4", scale_factor=0.6)
        
        # The line
        critical_line = Line(start=self.grid["C3"] + UP*0.5, end=self.grid["F3"] + DOWN*0.5, color="#FF0000", stroke_width=4)
        
        # Formula
        zeta_formula = MathTex(r"\zeta(s) = \sum_{n=1}^{\infty} n^{-s}", color="#FFFFFF")
        self.place_at_grid(zeta_formula, "B5", scale_factor=0.7)
        
        zeta_label = Text("Riemann Zeta Function", font_size=18, color="#FFFFFF")
        self.place_at_grid(zeta_label, "B4", scale_factor=0.75)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        self.play(FadeIn(strip))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        self.play(Create(critical_line))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        # Just maintain state, label line
        label = Text("Re(s) = 1/2", font_size=16, color="#FF0000")
        label.next_to(critical_line, RIGHT)
        self.play(Write(label))
        
        self.wait(2)
