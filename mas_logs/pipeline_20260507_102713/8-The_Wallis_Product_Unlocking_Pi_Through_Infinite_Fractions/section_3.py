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
        lecture_lines = [
            'Integration by parts gives us a recursive reduction formula.', 
            'We compute I sub n by stepping down two units.', 
            'Even paths start at Pi halves and descend regularly.', 
            'Odd paths begin at one and follow the pattern.', 
            'These two branches will soon merge into one formula.'
        ]
        self.setup_layout("The Recursive Pattern: The Reduction Formula", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Construct formula: I_n = ((n-1)/n) * I_{n-2}
        i_n = VGroup(Text("I", font_size=24), Text("n", font_size=16)).arrange(RIGHT, buff=0.02)
        i_n[1].shift(0.08*DOWN)
        
        equals = Text(" = ", font_size=24)
        
        num = Text("n-1", font_size=22)
        den = Text("n", font_size=22)
        frac_line = Line(LEFT, RIGHT).scale(0.35)
        fraction = VGroup(num, frac_line, den).arrange(DOWN, buff=0.08)
        
        i_nm2 = VGroup(Text("I", font_size=24), Text("n-2", font_size=16)).arrange(RIGHT, buff=0.02)
        i_nm2[1].shift(0.08*DOWN)
        
        formula = VGroup(i_n, equals, fraction, i_nm2).arrange(RIGHT, buff=0.15).set_color(WHITE)
        
        # Fix for Issue 26 & 27: Lower position and adjusted scale
        self.place_in_area(formula, "B2", "C5", scale_factor=0.9)
        
        box = SurroundingRectangle(formula, color=WHITE, buff=0.2)
        
        self.play(Write(formula), Create(box))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Highlight "step down by 2" with pulsing red bracket
        step_bracket = BraceBetweenPoints(
            i_nm2.get_bottom() + 0.2*DOWN,
            i_n.get_bottom() + 0.2*DOWN,
            color="#FF0000"
        )
        minus_two = Text("-2", font_size=18, color="#FF0000").next_to(step_bracket, DOWN, buff=0.1)
        
        self.play(GrowFromCenter(step_bracket), Write(minus_two))
        self.play(
            step_bracket.animate.scale(1.2), 
            minus_two.animate.scale(1.2), 
            run_time=0.4, 
            rate_func=there_and_back
        )
        self.play(
            step_bracket.animate.scale(1.2), 
            minus_two.animate.scale(1.2), 
            run_time=0.4, 
            rate_func=there_and_back
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        even_title = Text("Even Path", font_size=24, color="#FFD700")
        self.place_at_grid(even_title, "D2")
        
        even_seq = Text("I₄ → I₂ → I₀", font_size=22, color="#FFD700")
        self.place_at_grid(even_seq, "E2")
        
        # Result I0 = pi/2
        i0_sub = Text("0", font_size=16)
        even_res_text = VGroup(
            Text("I", font_size=24), 
            i0_sub, 
            Text(" = π/2", font_size=24)
        ).arrange(RIGHT, buff=0.05).set_color(WHITE)
        i0_sub.shift(0.08*DOWN)
        self.place_at_grid(even_res_text, "F2")
        
        self.play(Write(even_title))
        self.play(Write(even_seq))
        self.play(FadeIn(even_res_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        odd_title = Text("Odd Path", font_size=24, color="#00FA9A")
        self.place_at_grid(odd_title, "D5")
        
        odd_seq = Text("I₅ → I₃ → I₁", font_size=22, color="#00FA9A")
        self.place_at_grid(odd_seq, "E5")
        
        # Result I1 = 1
        i1_sub = Text("1", font_size=16)
        odd_res_text = VGroup(
            Text("I", font_size=24), 
            i1_sub, 
            Text(" = 1", font_size=24)
        ).arrange(RIGHT, buff=0.05).set_color(WHITE)
        i1_sub.shift(0.08*DOWN)
        self.place_at_grid(odd_res_text, "F5")
        
        self.play(Write(odd_title))
        self.play(Write(odd_seq))
        self.play(FadeIn(odd_res_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Briefly emphasize both paths
        self.play(
            Indicate(even_res_text, color="#FFD700"),
            Indicate(odd_res_text, color="#00FA9A")
        )
        self.wait(2)
