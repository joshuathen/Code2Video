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
        title_text = "Mathematical Symmetry and Verification"
        lecture_lines = [
            "Dual graphs swap the roles of vertices and faces.",
            "Both graphs share the exact same number of edges.",
            "This preserves Euler’s characteristic across the transformation."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors for consistency
        color_g = "#88CCEE"      # Cyan-ish for original graph
        color_dual = "#DDCC77"   # Sand-ish for dual graph
        color_hl1 = "#88CCEE"    # Matching Line 1
        color_hl2 = "#CC6677"    # Matching Line 2
        color_hl3 = "#AA88DD"    # Matching Line 3

        # === Animation for Lecture Line 1 ===
        # "Dual graphs swap the roles of vertices and faces."
        self.play(self.lecture[0].animate.set_color(color_hl1))

        # Table Headers - Issue 42: header_g moved to B3
        header_g = Text("Graph G", font_size=24, color=color_g)
        header_dual = Text("Dual G*", font_size=24, color=color_dual)
        self.place_at_grid(header_g, "B3")
        self.place_at_grid(header_dual, "B5")

        # Row Labels for V and F - Issue 41: Moved labels from col 1 to area col 1-2
        label_v = Text("Vertices (V)", font_size=22)
        label_f = Text("Faces (F)", font_size=22)
        self.place_in_area(label_v, "C1", "C2", scale_factor=0.8)
        self.place_in_area(label_f, "E1", "E2", scale_factor=0.8)

        # Values for G and G* (V and F) - Issue 42: val_g moved to col 3
        val_g_v = Text("5", font_size=28, color=color_g)
        val_g_f = Text("4", font_size=28, color=color_g)
        val_dual_v = Text("4", font_size=28, color=color_dual)
        val_dual_f = Text("5", font_size=28, color=color_dual)

        self.place_at_grid(val_g_v, "C3")
        self.place_at_grid(val_g_f, "E3")
        self.place_at_grid(val_dual_v, "C5")
        self.place_at_grid(val_dual_f, "E5")

        # Cross arrows for swap visualization
        # Updated start/end points to match column shifts
        arrow_v_to_f = CurvedArrow(self.grid["C3"] + DOWN*0.2, self.grid["E5"] + LEFT*0.3, angle=-TAU/6, color=color_hl1)
        arrow_f_to_v = CurvedArrow(self.grid["E3"] + DOWN*0.2, self.grid["C5"] + RIGHT*0.3, angle=TAU/6, color=color_hl1)

        self.play(FadeIn(header_g), FadeIn(header_dual))
        self.play(Write(label_v), Write(label_f))
        self.play(Write(val_g_v), Write(val_g_f))
        self.play(Write(val_dual_v), Write(val_dual_f))
        self.play(Create(arrow_v_to_f), Create(arrow_f_to_v))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Both graphs share the exact same number of edges."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color_hl2),
            Uncreate(arrow_v_to_f),
            Uncreate(arrow_f_to_v)
        )

        # Issue 41: label_e moved to area col 1-2
        label_e = Text("Edges (E)", font_size=22)
        self.place_in_area(label_e, "D1", "D2", scale_factor=0.8)

        # Issue 42: val_g_e moved to D3
        val_g_e = Text("7", font_size=28, color=color_g)
        val_dual_e = Text("7", font_size=28, color=color_dual)
        self.place_at_grid(val_g_e, "D3")
        self.place_at_grid(val_dual_e, "D5")

        self.play(Write(label_e))
        self.play(Write(val_g_e), Write(val_dual_e))
        
        # Equality sign - Issue 42: eq_sign moved to D4
        eq_sign = Text("=", font_size=28, color=color_hl2)
        self.place_at_grid(eq_sign, "D4")

        self.play(Write(eq_sign))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "This preserves Euler’s characteristic across the transformation."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color_hl3),
            FadeOut(eq_sign)
        )

        # Euler's Formula verification for both
        # Formula: V - E + F = 2
        formula_g = Text("5 - 7 + 4 = 2", font_size=24, t2c={"5": color_g, "7": color_g, "4": color_g, "2": WHITE})
        formula_dual = Text("4 - 7 + 5 = 2", font_size=24, t2c={"4": color_dual, "7": color_dual, "5": color_dual, "2": WHITE})
        
        # Issue 43: formula moved to areas for width handling
        self.place_in_area(formula_g, "F2", "F3", scale_factor=0.7)
        self.place_in_area(formula_dual, "F5", "F6", scale_factor=0.7)

        self.play(Write(formula_g), Write(formula_dual))

        # Verification Box Highlight
        rect_g = SurroundingRectangle(formula_g, color=color_hl3, buff=0.1)
        rect_dual = SurroundingRectangle(formula_dual, color=color_hl3, buff=0.1)
        
        self.play(Create(rect_g), Create(rect_dual))
        self.wait(2)
