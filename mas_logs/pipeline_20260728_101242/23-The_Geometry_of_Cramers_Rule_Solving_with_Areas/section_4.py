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
        # Define lecture lines based on storyboard
        lecture_lines = [
            "Let's replace the first vector v1 with b.",
            "This creates a new, shifted parallelogram.",
            "Its height remains relative to vector v2.",
            "The new area is scaled exactly by x.",
            "This is the key insight of Cramer's rule."
        ]
        self.setup_layout("The 'Shifted Parallelogram' Trick", lecture_lines)
        
        # Paths for assets
        vector_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/vector.svg"
        parallelogram_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/parallelogram.svg"

        # --- Build the Diagram ---
        # 1. Vectors (using SVG assets where requested)
        # origin is local (0,0,0)
        
        v1 = SVGMobject(vector_path).set_color("#FFFF00")
        v1.stretch_to_fit_width(1.5).stretch_to_fit_height(0.2)
        v1.move_to(0.75 * RIGHT) # Tail at local origin
        
        v2 = Arrow(ORIGIN, 1.5 * UP, buff=0, color="#0000FF")
        
        b_vec = np.array([3.0, 1.5, 0]) # x=2 (scaled by 1.5), y=1 (scaled by 1.5)
        b = SVGMobject(vector_path).set_color("#FF0000")
        b.stretch_to_fit_width(np.linalg.norm(b_vec)).stretch_to_fit_height(0.2)
        b.rotate(np.arctan2(b_vec[1], b_vec[0]), about_point=ORIGIN)
        b.move_to(b_vec / 2) # Tail at local origin
        
        v1_label = MathTex("v_1", color="#FFFF00", font_size=24).next_to(v1.get_right(), DOWN, buff=0.1)
        v2_label = MathTex("v_2", color="#0000FF", font_size=24).next_to(v2.get_top(), LEFT, buff=0.1)
        b_label = MathTex("b", color="#FF0000", font_size=24).next_to(b.get_top(), UR, buff=0.1)

        # 2. Parallelograms (using SVG assets)
        # Original yellow parallelogram
        poly1 = SVGMobject(parallelogram_path).set_color("#FFFF00").set_opacity(0.3)
        poly1.stretch_to_fit_width(1).stretch_to_fit_height(1)
        # Map unit square to (v1, v2) where v1=[1.5, 0], v2=[0, 1.5]
        poly1.apply_matrix([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1]])
        poly1.move_to(0.75 * RIGHT + 0.75 * UP)

        # New orange parallelogram
        poly_b = SVGMobject(parallelogram_path).set_color("#FFA500").set_opacity(0.3)
        poly_b.stretch_to_fit_width(1).stretch_to_fit_height(1)
        # Map unit square to (b, v2) where b=[3.0, 1.5], v2=[0, 1.5]
        # Matrix columns are the target vectors
        poly_b.apply_matrix([[3.0, 0, 0], [1.5, 1.5, 0], [0, 0, 1]])
        poly_b.move_to(1.5 * RIGHT + 1.5 * UP)

        # 3. Auxiliary lines for height (relative to local origin)
        base_line = Line(0.5 * DOWN, 3.0 * UP, color="#0000FF", stroke_width=1)
        h1_line = DashedLine(1.5 * RIGHT, [0, 0, 0], color="#FFFF00") # from v1 tip to origin
        hb_line = DashedLine(b_vec, [0, 1.5, 0], color="#FF0000") # from b tip to v2 axis

        # Group everything for global placement
        diagram_group = VGroup(v1, v2, b, v1_label, v2_label, b_label, poly1, poly_b, base_line, h1_line, hb_line)
        self.place_in_area(diagram_group, 'B3', 'E6', scale_factor=0.8)
        
        # Hide components for step-by-step introduction
        self.remove(v1, v2, b, v1_label, v2_label, b_label, poly1, poly_b, base_line, h1_line, hb_line)

        # === Animation for Lecture Line 1 ===
        # "Let's replace the first vector v1 with b."
        self.lecture[0].set_color("#FFFF00")
        self.play(FadeIn(v1), Create(v2), Write(v1_label), Write(v2_label), FadeIn(poly1), run_time=1.5)
        self.wait(1)
        # Morphing introduction of b
        self.play(
            FadeIn(b),
            Write(b_label),
            v1.animate.set_opacity(0.3),
            v1_label.animate.set_opacity(0.3),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "This creates a new, shifted parallelogram."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFA500")
        self.play(FadeIn(poly_b), poly1.animate.set_fill(opacity=0.05), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Its height remains relative to vector v2."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#0000FF")
        self.play(Create(base_line), Create(h1_line), run_time=1)
        self.wait(0.5)
        self.play(Transform(h1_line, hb_line), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "The new area is scaled exactly by x."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FFA500")
        self.play(poly_b.animate.set_fill(opacity=0.7), run_time=1)
        self.play(Flash(poly_b, color="#FFA500", line_length=0.3), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "This is the key insight of Cramer's rule."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFFF00")
        
        formula = MathTex(r"\text{Area}(b, v_2) = x \cdot \text{Area}(v_1, v_2)", color=WHITE, font_size=32)
        # VideoCritic fix: use place_in_area for formula
        self.place_in_area(formula, 'F2', 'F5', scale_factor=0.9)
        self.play(Write(formula), run_time=2)
        self.wait(2)
