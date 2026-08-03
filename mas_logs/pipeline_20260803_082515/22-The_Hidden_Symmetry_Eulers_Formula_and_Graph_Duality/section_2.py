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
        self.setup_layout("Euler’s Magic Number: V - E + F = 2", [
            "Every connected planar graph follows a mathematical invariant.",
            "Subtract edges from vertices, then add the faces.",
            "This alternating sum always equals exactly two.",
            "Add a vertex and edge; the sum remains constant.",
            "Euler's formula holds regardless of how the graph stretches."
        ])

        # Colors for highlighting
        COLOR_HIGHLIGHT = "#FFFF00"
        COLOR_V = "#88CCFF"
        COLOR_E = "#FF8888"
        COLOR_F = "#88FF88"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_HIGHLIGHT)
        
        # Define vertices
        v1_pos = self.grid["B2"]
        v2_pos = self.grid["B5"]
        v3_pos = self.grid["D3"]
        
        dot1 = Dot(v1_pos, color=COLOR_V)
        dot2 = Dot(v2_pos, color=COLOR_V)
        dot3 = Dot(v3_pos, color=COLOR_V)
        
        edge1 = Line(v1_pos, v2_pos, color=COLOR_E)
        edge2 = Line(v2_pos, v3_pos, color=COLOR_E)
        edge3 = Line(v3_pos, v1_pos, color=COLOR_E)
        
        graph = VGroup(edge1, edge2, edge3, dot1, dot2, dot3)
        
        # Labels for V, E, F - Addressing Issue 24 and 25
        v_label = MathTex("V=3", color=COLOR_V, font_size=32)
        self.place_at_grid(v_label, "A4") # Issue 24 fix
        
        e_label = MathTex("E=3", color=COLOR_E, font_size=32)
        self.place_at_grid(e_label, "C6") # Issue 25 fix
        
        f_label = MathTex("F=2", color=COLOR_F, font_size=32)
        self.place_at_grid(f_label, "C3") # Inside face
        
        self.play(Create(graph), FadeIn(v_label), FadeIn(e_label), FadeIn(f_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_HIGHLIGHT)
        
        # Formula: V - E + F - Addressing Issue 23
        formula = MathTex("V", "-", "E", "+", "F", "=", "2", color=WHITE, font_size=40)
        self.place_in_area(formula, "E2", "E5", scale_factor=1.2) # Issue 23 fix
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_HIGHLIGHT)
        
        # Show 3 - 3 + 2 = 2 - Addressing Issue 23
        formula_nums = MathTex("3", "-", "3", "+", "2", "=", "2", color=WHITE, font_size=40)
        self.place_in_area(formula_nums, "E2", "E5", scale_factor=1.2) # Issue 23 fix
        
        self.play(ReplacementTransform(formula, formula_nums))
        self.play(Indicate(formula_nums[-1], color=COLOR_HIGHLIGHT))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_HIGHLIGHT)
        
        # Add an edge splitting the triangle (Face split: V=3, E=4, F=3)
        # Point on edge2 (v2-v3)
        mid_edge2 = (v2_pos + v3_pos) / 2
        edge4 = Line(v1_pos, mid_edge2, color=COLOR_E)
        
        # Update Labels
        e_label_new = MathTex("E=4", color=COLOR_E, font_size=32)
        self.place_at_grid(e_label_new, "C6")
        
        f_label_new = MathTex("F=3", color=COLOR_F, font_size=32)
        self.place_at_grid(f_label_new, "C3")
        
        # Update Calculation - Addressing Issue 23
        formula_nums_2 = MathTex("3", "-", "4", "+", "3", "=", "2", color=COLOR_HIGHLIGHT, font_size=40)
        self.place_in_area(formula_nums_2, "E2", "E5", scale_factor=1.2) # Issue 23 fix
        
        self.play(
            Create(edge4),
            Transform(e_label, e_label_new),
            Transform(f_label, f_label_new),
            ReplacementTransform(formula_nums, formula_nums_2)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_HIGHLIGHT)
        
        # Add a vertex and edge (V=4, E=5, F=3)
        # Split edge3 into two at a new vertex
        v4_pos = (v3_pos + v1_pos) / 2
        dot4 = Dot(v4_pos, color=COLOR_V)
        
        # Visual split: hide edge3, add edge3a, edge3b
        edge3.set_alpha(0)
        edge3a = Line(v3_pos, v4_pos, color=COLOR_E)
        edge3b = Line(v4_pos, v1_pos, color=COLOR_E)
        
        # Update labels
        v_label_new_val = MathTex("V=4", color=COLOR_V, font_size=32)
        self.place_at_grid(v_label_new_val, "A4")
        
        e_label_final_val = MathTex("E=5", color=COLOR_E, font_size=32)
        self.place_at_grid(e_label_final_val, "C6")
        
        # Update Calculation - Addressing Issue 23
        formula_nums_3 = MathTex("4", "-", "5", "+", "3", "=", "2", color=COLOR_HIGHLIGHT, font_size=40)
        self.place_in_area(formula_nums_3, "E2", "E5", scale_factor=1.2) # Issue 23 fix

        self.play(
            FadeIn(dot4),
            Create(edge3a),
            Create(edge3b),
            Transform(v_label, v_label_new_val),
            Transform(e_label, e_label_final_val),
            ReplacementTransform(formula_nums_2, formula_nums_3)
        )
        
        # Stretch animation to show invariance
        self.play(
            dot3.animate.shift(DOWN*0.5 + RIGHT*0.2),
            dot4.animate.shift(DOWN*0.25 + RIGHT*0.1),
            edge2.animate.put_start_and_end_on(v2_pos, v3_pos + DOWN*0.5 + RIGHT*0.2),
            edge3a.animate.put_start_and_end_on(v3_pos + DOWN*0.5 + RIGHT*0.2, v4_pos + DOWN*0.25 + RIGHT*0.1),
            edge3b.animate.put_start_and_end_on(v4_pos + DOWN*0.25 + RIGHT*0.1, v1_pos),
            edge4.animate.put_start_and_end_on(v1_pos, (v2_pos + v3_pos + DOWN*0.5 + RIGHT*0.2) / 2),
            run_time=2
        )
        self.wait(2)
