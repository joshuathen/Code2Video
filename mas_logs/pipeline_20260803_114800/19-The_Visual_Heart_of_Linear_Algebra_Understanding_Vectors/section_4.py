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
        lecture_lines = [
            "Adding vectors is like following sequential instructions.",
            "Place the second vector's tail at the first's tip.",
            "The result is the arrow from start to end.",
            "Numerically, we simply add the corresponding components.",
            "This new vector represents the combined total movement."
        ]
        self.setup_layout("Vector Addition: The Tip-to-Tail Method", lecture_lines)

        # Colors
        COLOR_VEC_A = "#1E90FF"
        COLOR_VEC_B = "#FF00FF"
        COLOR_RESULTANT = "#FFFF00"
        HIGHLIGHT_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Focus on the process of tip-to-tail
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Grid positions for vectors
        # Vector A: D2 to D4 (horizontal right)
        # Vector B: D4 to B4 (vertical up)
        start_point = self.grid["D2"]
        mid_point = self.grid["D4"]
        end_point = self.grid["B4"]
        
        # Vector A
        vec_a = Arrow(start_point, mid_point, buff=0, color=COLOR_VEC_A)
        label_a = Text("A", font_size=24, color=COLOR_VEC_A)
        self.place_at_grid(label_a, "E3", scale_factor=0.8)
        
        # Vector B
        vec_b = Arrow(mid_point, end_point, buff=0, color=COLOR_VEC_B)
        label_b = Text("B", font_size=24, color=COLOR_VEC_B)
        self.place_at_grid(label_b, "C5", scale_factor=0.8)
        
        self.play(Create(vec_a), Write(label_a))
        self.wait(0.5)
        self.play(Create(vec_b), Write(label_b))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Resultant vector from start of A to end of B
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_RESULTANT)
        )
        
        res_vec = Arrow(start_point, end_point, buff=0, color=COLOR_RESULTANT)
        res_label = Text("Resultant", font_size=24, color=COLOR_RESULTANT)
        # Fix for Issue 27: Position at C3 instead of C2
        self.place_at_grid(res_label, "C3", scale_factor=0.8)
        
        self.play(Create(res_vec), Write(res_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Numerical addition shown at bottom
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Displacement values: A is [2, 0], B is [0, 2]
        math_addition = MathTex(
            r"\begin{bmatrix} 2 \\ 0 \end{bmatrix}", 
            "+", 
            r"\begin{bmatrix} 0 \\ 2 \end{bmatrix}", 
            "=", 
            r"\begin{bmatrix} 2 \\ 2 \end{bmatrix}",
            font_size=36
        )
        math_addition[0].set_color(COLOR_VEC_A)
        math_addition[2].set_color(COLOR_VEC_B)
        math_addition[4].set_color(COLOR_RESULTANT)
        
        # Fix for Issue 28: scale factor 0.8
        self.place_in_area(math_addition, "F2", "F5", scale_factor=0.8)
        
        self.play(Write(math_addition))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Conclude with a visual flash on the result
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        self.play(Flash(res_vec, color=COLOR_RESULTANT, line_length=0.3, flash_radius=0.4))
        self.wait(2)
