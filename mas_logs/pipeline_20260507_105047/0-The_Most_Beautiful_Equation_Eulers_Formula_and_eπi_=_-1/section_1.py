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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            'Behold the most beautiful equation in all of mathematics.',
            'It unites five fundamental constants in a single identity.',
            'Meet zero, one, e, i, and pi.',
            'Geometry, calculus, and algebra converge in this expression.',
            "Let's explore how these different worlds become one."
        ]
        self.setup_layout("The Grand Reunion: Introduction to the Identity", lecture_lines)
        
        # Define equation parts to allow individual color/scale animations
        # Using Text instead of MathTex to avoid LaTeX dependencies
        eq_e = Text("e", color=WHITE)
        eq_i = Text("i", color=WHITE, font_size=20)
        eq_pi = Text("π", color=WHITE, font_size=20)
        eq_plus = Text("+", color=WHITE)
        eq_one = Text("1", color=WHITE)
        eq_eq = Text("=", color=WHITE)
        eq_zero = Text("0", color=WHITE)
        
        # Manual positioning for e^{iπ} + 1 = 0 look
        eq_i.next_to(eq_e, UR, buff=0.05).shift(UP*0.1)
        eq_pi.next_to(eq_i, RIGHT, buff=0.05)
        eq_plus.next_to(eq_e, RIGHT, buff=0.8)
        eq_one.next_to(eq_plus, RIGHT, buff=0.2)
        eq_eq.next_to(eq_one, RIGHT, buff=0.2)
        eq_zero.next_to(eq_eq, RIGHT, buff=0.2)
        
        equation = VGroup(eq_e, eq_i, eq_pi, eq_plus, eq_one, eq_eq, eq_zero)
        
        # Resolved Issue 32, 33, 34: Improved positioning and scaling
        self.place_in_area(equation, 'C2', 'D5', scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        # Behold the most beautiful equation...
        self.play(FadeIn(equation))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # It unites five fundamental constants...
        self.play(self.lecture[1].animate.set_color(WHITE))
        self.play(equation.animate.scale(1.1), run_time=0.5)
        self.play(equation.animate.scale(1/1.1), run_time=0.5)
        
        # === Animation for Lecture Line 3 ===
        # Meet zero, one, e, i, and pi. Highlight '0' and '1' Blue
        self.play(self.lecture[2].animate.set_color(BLUE))
        self.play(
            eq_zero.animate.set_color(BLUE).scale(1.5),
            eq_one.animate.set_color(BLUE).scale(1.5),
            run_time=0.8
        )
        self.play(
            eq_zero.animate.scale(1/1.5),
            eq_one.animate.scale(1/1.5),
            run_time=0.8
        )
        
        # === Animation for Lecture Line 4 ===
        # Geometry, calculus, and algebra... Highlight 'pi' Green
        self.play(self.lecture[3].animate.set_color(GREEN))
        self.play(
            eq_pi.animate.set_color(GREEN).scale(1.5),
            run_time=0.8
        )
        self.play(
            eq_pi.animate.scale(1/1.5),
            run_time=0.8
        )
        
        # === Animation for Lecture Line 5 ===
        # Let's explore how these different worlds... Highlight 'e' and 'i' Yellow
        self.play(self.lecture[4].animate.set_color(YELLOW))
        self.play(
            eq_e.animate.set_color(YELLOW).scale(1.5),
            eq_i.animate.set_color(YELLOW).scale(1.5),
            run_time=0.8
        )
        self.play(
            eq_e.animate.scale(1/1.5),
            eq_i.animate.scale(1/1.5),
            run_time=0.8
        )
        
        # Glow effect and fade out
        glow_rect = SurroundingRectangle(equation, color=WHITE, buff=0.2).set_stroke(width=0).set_fill(WHITE, opacity=0.2)
        self.play(FadeIn(glow_rect))
        self.play(
            FadeOut(equation),
            FadeOut(glow_rect),
            run_time=1.5
        )
        self.wait(1)
