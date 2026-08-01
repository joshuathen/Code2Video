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
        lecture_lines_text = [
            "A vector space is a set following specific axioms.",
            "Closure means results never leave the original set.",
            "Adding polynomials always results in another polynomial.",
            "Scaling integers can result in non-integers, failing closure.",
            "A valid vector space must include a zero vector."
        ]
        self.setup_layout("Defining the 'Vector Space Club' (The Axioms)", lecture_lines_text)

        # === Animation for Lecture Line 1 ===
        # Using Text with italic slant to mimic math symbols since LaTeX is unavailable
        self.lecture[0].set_color(WHITE)
        club_circle = Circle(radius=2.2, color=WHITE)
        self.place_in_area(club_circle, "A2", "F5")
        
        club_label = Text("Vector Space Club", font_size=24, color=WHITE)
        self.place_at_grid(club_label, "A3", scale_factor=0.8)
        club_label.shift(UP * 0.3)
        
        v_sym = Text("v", slant=ITALIC, color=WHITE)
        w_sym = Text("w", slant=ITALIC, color=WHITE)
        self.place_at_grid(v_sym, "B2", scale_factor=1.2)
        self.place_at_grid(w_sym, "D5", scale_factor=1.2)
        
        self.play(Create(club_circle), Write(club_label))
        self.play(FadeIn(v_sym), FadeIn(w_sym))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        sum_vw = Text("v + w", slant=ITALIC, color="#00FF00")
        kv_sym = Text("k · v", slant=ITALIC, color="#00FF00")
        
        self.place_at_grid(sum_vw, "C3", scale_factor=1.0)
        self.place_at_grid(kv_sym, "C4", scale_factor=1.0)
        
        arrow_sum = Arrow(v_sym.get_bottom(), sum_vw.get_top(), color="#00FF00", buff=0.1)
        arrow_kv = Arrow(v_sym.get_right(), kv_sym.get_left(), color="#00FF00", buff=0.1)
        
        self.play(
            FadeIn(sum_vw), FadeIn(kv_sym),
            Create(arrow_sum), Create(arrow_kv)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        poly1 = Text("x²", slant=ITALIC, color="#00FFFF")
        poly2 = Text("3x", slant=ITALIC, color="#00FFFF")
        poly_sum = Text("x² + 3x", slant=ITALIC, color="#00FFFF")
        
        self.place_at_grid(poly1, "B4", scale_factor=1.0)
        self.place_at_grid(poly2, "B5", scale_factor=1.0)
        self.place_at_grid(poly_sum, "D3", scale_factor=1.0)
        
        self.play(FadeIn(poly1), FadeIn(poly2))
        self.play(ReplacementTransform(VGroup(poly1.copy(), poly2.copy()), poly_sum))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF00FF")
        int_1 = Text("1", color="#FF00FF")
        scaled_int = Text("1 × 0.5 = 0.5", color="#FF00FF")
        red_x = Cross(scaled_int, stroke_color="#FF0000")
        
        self.place_at_grid(int_1, "B6", scale_factor=1.0)
        self.place_at_grid(scaled_int, "C6", scale_factor=0.8)
        
        self.play(FadeIn(int_1))
        self.play(ReplacementTransform(int_1.copy(), scaled_int))
        self.play(Create(red_x))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        zero_vec = Text("0", weight=BOLD, color=WHITE).scale(2.0)
        self.place_in_area(zero_vec, "C3", "D4")
        
        zero_label = Text("The Zero Vector", font_size=16, color="#FFD700")
        zero_label.next_to(zero_vec, DOWN, buff=0.1)
        
        self.play(FadeIn(zero_vec), Write(zero_label))
        self.play(Flash(zero_vec, color="#FFD700", line_length=0.3, num_lines=12))
        self.play(Indicate(zero_vec, color="#FFD700"))
        
        # Checklist
        checklist_title = Text("8 Axioms", font_size=18, color=WHITE)
        self.place_at_grid(checklist_title, "E1", scale_factor=1.0)
        
        axiom_list = VGroup(
            Text("1. Closure (+)", font_size=14, color=GREEN),
            Text("2. Closure (*)", font_size=14, color=GREEN),
            Text("3. Zero Vector", font_size=14, color=GREEN),
            Text("...", font_size=14, color=WHITE)
        ).arrange(DOWN, aligned_edge=LEFT)
        self.place_at_grid(axiom_list, "F1", scale_factor=1.0)
        axiom_list.next_to(checklist_title, DOWN, buff=0.1)
        
        self.play(FadeIn(checklist_title), FadeIn(axiom_list))
        self.wait(2)
