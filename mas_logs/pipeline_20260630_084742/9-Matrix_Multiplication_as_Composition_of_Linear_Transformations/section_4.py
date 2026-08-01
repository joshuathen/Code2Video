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
        # Setup Title and Lecture lines
        title_text = "The Math: Why Right-to-Left?"
        lecture_lines = [
            "Matrix notation works from right to left.",
            "The matrix closest to the vector acts first.",
            "We combine these matrices into one single operation."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Color constants
        COLOR_B = "#00FFFF"  # Cyan
        COLOR_A = "#FFFF00"  # Yellow
        COLOR_X = "#FFFFFF"  # White

        # === Animation for Lecture Line 1 ===
        # Update lecture color
        self.play(self.lecture[0].animate.set_color(COLOR_B))

        # Display expression B(A(x)) using Text to avoid LaTeX dependency
        # Indices: 0:B, 1:(, 2:A, 3:(, 4:x, 5:), 6:)
        expr_bax = Text("B(A(x))", font_size=60)
        expr_bax[0].set_color(COLOR_B)
        expr_bax[2].set_color(COLOR_A)
        expr_bax[4].set_color(COLOR_X)
        
        # Position expression in top half of the grid
        self.place_in_area(expr_bax, "B1", "C6", scale_factor=1.0)
        self.play(Write(expr_bax))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Update lecture color
        self.play(self.lecture[1].animate.set_color(COLOR_A))

        # Highlight 'A' and 'x' to show they are evaluated first
        # Indices 2, 3, 4, 5 represent "A(x)"
        box_ax = SurroundingRectangle(expr_bax[2:6], color=COLOR_A, buff=0.1)
        
        # Explanatory text for evaluation order
        order_text = Text("Evaluated First", font_size=18, color=COLOR_A)
        self.place_in_area(order_text, "D1", "D6", scale_factor=0.8)
        
        # Visual cue: arrow indicating direction
        arrow = Arrow(start=self.grid["B6"], end=self.grid["B1"], color=WHITE, buff=0.1)
        self.play(Create(box_ax), Write(order_text), Create(arrow))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Update lecture color
        self.play(self.lecture[2].animate.set_color(WHITE))

        # Show B * A fusing into C
        # Indices: 0:B, 1:·, 2:A
        expr_fusion_base = Text("B·A", font_size=60)
        expr_fusion_base[0].set_color(COLOR_B)
        expr_fusion_base[2].set_color(COLOR_A)
        self.place_in_area(expr_fusion_base, "E1", "F6", scale_factor=1.0)
        
        self.play(Write(expr_fusion_base))
        self.wait(0.5)

        # Morph B * A into a single symbol 'C'
        expr_c_result = Text("C", font_size=60, color=WHITE)
        self.place_in_area(expr_c_result, "E1", "F6", scale_factor=1.0)
        
        # Also update the top expression to C(x)
        # Indices: 0:C, 1:(, 2:x, 3:)
        expr_cx = Text("C(x)", font_size=60)
        expr_cx[0].set_color(WHITE)
        expr_cx[2].set_color(COLOR_X)
        self.place_in_area(expr_cx, "B1", "C6", scale_factor=1.0)

        self.play(
            ReplacementTransform(expr_fusion_base, expr_c_result),
            ReplacementTransform(expr_bax, expr_cx),
            FadeOut(box_ax),
            FadeOut(order_text),
            FadeOut(arrow)
        )
        self.wait(3)
