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
        title_str = "Vector Addition: The Tip-to-Tail Method"
        lecture_lines = [
            "To add vectors, use the tip-to-tail method.",
            "Place the second vector's tail at the first's tip.",
            "The resultant vector connects the start to the end.",
            "Algebraically, we sum their horizontal and vertical components.",
            "This new vector represents the combined total movement."
        ]
        self.setup_layout(title_str, lecture_lines)

        # Define Colors
        color_a = YELLOW
        color_b = BLUE
        color_r = "#FF0000" # Red

        # Create objects (persistent)
        # Vector A: [4, 0] from E2 to E6
        vec_a = Arrow(self.grid["E2"], self.grid["E6"], buff=0, color=color_a)
        # Resolved Issue 31: Place label_a at F3 instead of F4
        label_a = MathTex(r"\vec{A} = [4, 0]", color=color_a)
        self.place_at_grid(label_a, "F3", scale_factor=0.6)
        
        # Vector B: [0, 3] from E2 to B2
        vec_b = Arrow(self.grid["E2"], self.grid["B2"], buff=0, color=color_b)
        # Resolved Issue 30: Initially place label_b at B2 (tip of vec_b)
        # It will shift to B6 (the final tip) during the animation.
        label_b = MathTex(r"\vec{B} = [0, 3]", color=color_b)
        self.place_at_grid(label_b, "B2", scale_factor=0.6)
        
        # Resultant Vector: [4, 3] from E2 to B6
        vec_r = Arrow(self.grid["E2"], self.grid["B6"], buff=0, color=color_r)
        # Resolved Issue 32: Place label_r at B3 instead of C4 to avoid overlap
        label_r = MathTex(r"\vec{R} = [4, 3]", color=color_r)
        self.place_at_grid(label_r, "B3", scale_factor=0.6)
        
        # Math Equation
        math_sum = MathTex(
            r"\vec{R} = \vec{A} + \vec{B} = \begin{bmatrix} 4 \\ 0 \end{bmatrix} + \begin{bmatrix} 0 \\ 3 \end{bmatrix} = \begin{bmatrix} 4 \\ 3 \end{bmatrix}",
            color=WHITE
        )
        self.place_in_area(math_sum, "A3", "A6", scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        # "To add vectors, use the tip-to-tail method."
        self.play(self.lecture[0].animate.set_color(color_a))
        self.play(Create(vec_a), Write(label_a), Create(vec_b), Write(label_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Place the second vector's tail at the first's tip."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color_b)
        )
        
        # Shift B to tip of A (E6). E6 - E2 is the displacement of A.
        shift_vector = self.grid["E6"] - self.grid["E2"]
        self.play(
            vec_b.animate.shift(shift_vector),
            label_b.animate.shift(shift_vector)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The resultant vector connects the start to the end."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color_r)
        )
        self.play(Create(vec_r), Write(label_r))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Algebraically, we sum their horizontal and vertical components."
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(WHITE)
        )
        self.play(Write(math_sum))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "This new vector represents the combined total movement."
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(color_r)
        )
        # Final highlight
        self.play(
            vec_r.animate.set_stroke_width(10),
            run_time=0.5
        )
        self.play(
            vec_r.animate.set_stroke_width(4),
            run_time=0.5
        )
        self.wait(2)
