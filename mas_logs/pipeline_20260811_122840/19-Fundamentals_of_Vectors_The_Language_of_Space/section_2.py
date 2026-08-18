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
        self.setup_layout("Visualizing Vector Addition", ["Place vectors tip-to-tail.", "The resultant connects start to finish.", "Addition is commutative."])
        
        # Load SVG Asset
        asset_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        self.place_at_grid(asset_icon, 'A6', scale_factor=0.3)
        self.add(asset_icon)
        
        # Colors: #00FFFF, #FF00FF, #FFFF00
        # Mobjects
        u = Arrow(start=ORIGIN, end=RIGHT*1.5, color="#00FFFF")
        v = Arrow(start=ORIGIN, end=UP*1.2, color="#FF00FF")
        
        label_u = MathTex("u", color="#00FFFF").scale(0.7)
        label_v = MathTex("v", color="#FF00FF").scale(0.7)
        
        # Placement using requested grid positions (Fixes issues 23, 24, 36)
        self.place_at_grid(u, 'D2', scale_factor=0.8)
        label_u.next_to(u, DOWN, buff=0.1)
        
        # Use D3 to keep proximity
        self.place_at_grid(v, 'D3', scale_factor=0.8)
        label_v.next_to(v, RIGHT, buff=0.1)

        # === Animation for Lecture Line 1 ===
        # Show vectors u and v head-to-tail
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.play(Create(u), Write(label_u))
        
        v_shifted = v.copy().shift(u.get_end() - ORIGIN)
        label_v_shifted = label_v.copy().next_to(v_shifted, UP, buff=0.1)
        
        self.play(Create(v_shifted), Write(label_v_shifted))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The resultant connects start to finish.
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        # Resultant from start of u to end of v_shifted
        resultant = Arrow(start=u.get_start(), end=v_shifted.get_end(), color="#FFFF00")
        resultant_formula = MathTex("u + v", color="#FFFF00")
        
        # Fixes encroachment/anchoring (Issue 25, 36)
        self.place_in_area(resultant_formula, 'C2', 'C4', scale_factor=0.7)
        
        self.play(Create(resultant), Write(resultant_formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Addition is commutative.
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        
        # Show v then u
        v_first = v.copy().move_to(self.grid['D2'])
        u_shifted_after_v = u.copy().shift(v_first.get_end() - ORIGIN)
        
        self.play(FadeOut(u), FadeOut(v_shifted), FadeOut(label_u), FadeOut(label_v_shifted), FadeOut(resultant), FadeOut(resultant_formula))
        self.play(Create(v_first), Create(u_shifted_after_v))
        self.wait(2)
