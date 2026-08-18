from manim import *
import numpy as np

# === TeachingScene Base Class ===
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

class Section1Scene(TeachingScene):
    def construct(self):
        # Section 1: Prerequisite: Beyond the Arrow
        title_str = "Prerequisite: Beyond the Arrow"
        lines_str = [
            "Vectors are usually seen as arrows with direction.",
            "But we can view them simply as coordinate lists.",
            "These objects must support basic addition and scaling."
        ]
        self.setup_layout(title_str, lines_str)
        
        # Colors
        COLOR_VEC = "#00FF00"  # Green for first two lines and their assets
        COLOR_MATH = "#58C4DD" # Blue-ish for operations
        
        # === Animation for Lecture Line 1 ===
        # "Vectors are usually seen as arrows with direction."
        self.play(self.lecture[0].animate.set_color(COLOR_VEC))
        
        # Visual: A green arrow
        arrow_visual = Vector([2, 1], color=COLOR_VEC)
        self.place_in_area(arrow_visual, "B2", "E4", scale_factor=1.2)
        
        self.play(GrowArrow(arrow_visual))
        self.wait(1.5)
        
        # === Animation for Lecture Line 2 ===
        # "But we can view them simply as coordinate lists."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_VEC)
        )
        
        # Visual: Coordinate list [x, y]
        coord_list = MathTex(r"\begin{bmatrix} x \\ y \end{bmatrix}", color=COLOR_VEC)
        self.place_at_grid(coord_list, "C5", scale_factor=1.5)
        
        # Transform arrow to coordinate list
        self.play(Transform(arrow_visual, coord_list))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # "These objects must support basic addition and scaling."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_MATH)
        )
        
        # Clear previous transformation
        self.play(FadeOut(arrow_visual))
        
        # 3.1 Scaling Visual Formula
        scaling_eq = MathTex(r"s \cdot \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} sx \\ sy \end{bmatrix}", color=COLOR_MATH)
        self.place_at_grid(scaling_eq, "B3", scale_factor=0.8)
        
        # 3.2 Addition Visual Formula
        addition_eq = MathTex(r"\begin{bmatrix} x_1 \\ y_1 \end{bmatrix} + \begin{bmatrix} x_2 \\ y_2 \end{bmatrix} = \begin{bmatrix} x_1+x_2 \\ y_1+y_2 \end{bmatrix}", color=COLOR_MATH)
        self.place_at_grid(addition_eq, "D3", scale_factor=0.7)
        
        # 3.3 Tip-to-tail visualization using grid points
        p_start = self.grid["E4"]
        p_mid = self.grid["D5"]
        p_end = self.grid["B5"]
        
        vec_1 = Arrow(p_start, p_mid, color=BLUE, buff=0)
        vec_2 = Arrow(p_mid, p_end, color=RED, buff=0)
        vec_sum = Arrow(p_start, p_end, color=YELLOW, buff=0)
        
        label_v1 = MathTex(r"\vec{v}_1", color=BLUE)
        label_v2 = MathTex(r"\vec{v}_2", color=RED)
        self.place_at_grid(label_v1, "F4", scale_factor=0.7)
        self.place_at_grid(label_v2, "C6", scale_factor=0.7)

        self.play(Write(scaling_eq))
        self.wait(0.5)
        self.play(Write(addition_eq))
        self.wait(1)
        
        self.play(Create(vec_1), FadeIn(label_v1))
        self.play(Create(vec_2), FadeIn(label_v2))
        self.wait(0.5)
        self.play(Create(vec_sum))
        
        self.wait(3)
        
        # End sequence
        self.play(
            FadeOut(scaling_eq), FadeOut(addition_eq),
            FadeOut(vec_1), FadeOut(vec_2), FadeOut(vec_sum),
            FadeOut(label_v1), FadeOut(label_v2),
            self.lecture[2].animate.set_color(WHITE)
        )
        self.wait(1)
