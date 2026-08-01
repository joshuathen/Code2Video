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
        # Setup layout
        title_text = "Defining Eigenvalues: The Scale of Change"
        lecture_lines = [
            "Eigenvalues measure how much eigenvectors stretch or shrink.",
            "If the whisker triples in length, lambda is three.",
            "Negative values mean the vector flips direction."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Vector styling
        EIGENVECTOR_COLOR = "#52C41A"
        SCALAR_COLOR = "#F5222D"
        
        # === Animation for Lecture Line 1 ===
        # Line 1: Eigenvalues measure how much eigenvectors stretch or shrink.
        self.lecture[0].set_color(YELLOW)
        
        # Base eigenvector (unit length)
        # Using grid points: D2 is x=1.5, D3 is x=2.5. Vector points from D2 to D3 (length 1).
        v_start = self.grid["D2"]
        v_end_1 = self.grid["D3"]
        vec = Arrow(v_start, v_end_1, color=EIGENVECTOR_COLOR, buff=0, stroke_width=6)
        
        # Labeling the vector with Text "v"
        vec_label = Text("v", color=EIGENVECTOR_COLOR, font_size=24, slant=ITALIC)
        # Fixed Issue 31: Moving label from E2 to E3
        self.place_at_grid(vec_label, "E3", scale_factor=0.8)
        
        self.play(GrowArrow(vec), FadeIn(vec_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: If the whisker triples in length, lambda is three.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Scale the vector from length 1 to length 3
        # D2 is x=1.5. D5 is x=4.5. Distance = 3.0.
        v_end_3 = self.grid["D5"]
        vec_3 = Arrow(v_start, v_end_3, color=EIGENVECTOR_COLOR, buff=0, stroke_width=6)
        
        # Label λ = 3 (Using Unicode Lambda)
        lambda_3_label = Text("λ = 3", color=SCALAR_COLOR, font_size=32)
        self.place_at_grid(lambda_3_label, "C4")
        
        # Equation Av = 3v constructed via Text VGroup
        eq_parts = VGroup(
            Text("A", color=WHITE, font_size=28, weight=BOLD),
            Text("v", color=EIGENVECTOR_COLOR, font_size=28, slant=ITALIC),
            Text(" = ", color=WHITE, font_size=28),
            Text("3", color=SCALAR_COLOR, font_size=32),
            Text("v", color=EIGENVECTOR_COLOR, font_size=28, slant=ITALIC)
        ).arrange(RIGHT, buff=0.1)
        self.place_at_grid(eq_parts, "B4")
        
        # Issue 22: Integrate Whis asset
        whis = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/whis.svg")
        self.place_at_grid(whis, "B5", scale_factor=0.6)
        
        self.play(
            ReplacementTransform(vec, vec_3),
            FadeIn(lambda_3_label),
            FadeIn(eq_parts),
            FadeIn(whis)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Line 3: Negative values mean the vector flips direction.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Flip the vector: Point to x=0.5 (D1). Start is x=1.5 (D2). Length is 1, but flipped.
        v_end_neg = self.grid["D1"]
        vec_neg = Arrow(v_start, v_end_neg, color=EIGENVECTOR_COLOR, buff=0, stroke_width=6)
        
        # Label λ = -1
        lambda_neg_label = Text("λ = -1", color=SCALAR_COLOR, font_size=32)
        # Fixed Issue 30: Moving lambda label from C1 to C4 for consistency
        self.place_at_grid(lambda_neg_label, "C4")
        
        # Update equation to Av = -1v
        eq_neg_parts = VGroup(
            Text("A", color=WHITE, font_size=28, weight=BOLD),
            Text("v", color=EIGENVECTOR_COLOR, font_size=28, slant=ITALIC),
            Text(" = ", color=WHITE, font_size=28),
            Text("-1", color=SCALAR_COLOR, font_size=32),
            Text("v", color=EIGENVECTOR_COLOR, font_size=28, slant=ITALIC)
        ).arrange(RIGHT, buff=0.1)
        # Fixed Issue 29: Moving equation from B1 to B4 for consistency
        self.place_at_grid(eq_neg_parts, "B4")
        
        # Move Whis slightly to point at the new relationship if needed, 
        # but B5 remains a good relative position.
        
        self.play(
            ReplacementTransform(vec_3, vec_neg),
            ReplacementTransform(lambda_3_label, lambda_neg_label),
            ReplacementTransform(eq_parts, eq_neg_parts)
        )
        self.wait(3)

        # Reset final color
        self.lecture[2].set_color(WHITE)
        self.wait(1)
