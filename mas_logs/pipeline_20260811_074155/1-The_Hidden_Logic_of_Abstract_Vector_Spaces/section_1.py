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
        # Section title and lecture lines
        title_str = "Prerequisite: Beyond the Arrow"
        lecture_lines = [
            "Vectors are usually seen as arrows with direction.",
            "But we can view them simply as coordinate lists.",
            "These objects must support basic addition and scaling."
        ]
        self.setup_layout(title_str, lecture_lines)

        # Colors for the theme
        COLOR_VEC = "#00FF00"  # Green
        COLOR_MATH = "#58C4DD" # Blue-ish
        COLOR_HL = "#FFFF00"   # Yellow
        
        # === Animation for Lecture Line 1 ===
        # "Vectors are usually seen as arrows with direction."
        self.play(self.lecture[0].animate.set_color(COLOR_VEC))
        
        # Show an arrow with magnitude/direction labeling
        # Use Arrow with grid-based points for consistency
        v_start = self.grid["C2"]
        v_end = self.grid["B3"]
        v_arrow = Arrow(v_start, v_end, color=COLOR_VEC, buff=0)
        
        v_label = Text("Mag & Dir", font_size=16, color=COLOR_VEC)
        # Fix Issue 19: Reposition v_label
        self.place_in_area(v_label, 'C3', 'D4', scale_factor=0.8)
        
        self.play(GrowArrow(v_arrow), FadeIn(v_label))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # "But we can view them simply as coordinate lists."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_VEC)
        )
        
        # Represent as a coordinate list
        coord_tex = MathTex(r"\begin{bmatrix} x \\ y \end{bmatrix}", color=COLOR_VEC)
        self.place_at_grid(coord_tex, "C5", scale_factor=1.2)
        
        # Demonstrate scaling conceptually
        scaling_tex = MathTex(r"2 \cdot \begin{bmatrix} x \\ y \end{bmatrix}", color=COLOR_HL)
        # Fix Issue 17: Reposition scaling_tex and replace coord_tex to avoid clutter
        self.place_at_grid(scaling_tex, 'C5', scale_factor=1.2)
        
        # Scaled arrow visual
        v_stretched_start = self.grid["C2"]
        v_stretched_end = self.grid["A4"]
        v_stretched = Arrow(v_stretched_start, v_stretched_end, color=COLOR_VEC, buff=0)
        
        self.play(
            ReplacementTransform(v_arrow.copy(), coord_tex),
            FadeOut(v_label)
        )
        self.wait(1)
        
        # Transition from coordinate list to scaled version
        self.play(ReplacementTransform(coord_tex, scaling_tex))
        self.play(ReplacementTransform(v_arrow, v_stretched))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # "These objects must support basic addition and scaling."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_MATH),
            FadeOut(scaling_tex),
            FadeOut(v_stretched)
        )
        
        # Vector addition visual (tip-to-tail)
        p_start = self.grid["E4"]
        p_mid = self.grid["D5"]
        p_end = self.grid["B6"]
        
        vec1 = Arrow(p_start, p_mid, color=BLUE, buff=0)
        vec2 = Arrow(p_mid, p_end, color=RED, buff=0)
        vec_s = Arrow(p_start, p_end, color=COLOR_MATH, buff=0)
        
        lbl1 = MathTex(r"v_1", color=BLUE)
        lbl2 = MathTex(r"v_2", color=RED)
        self.place_at_grid(lbl1, "F4", scale_factor=0.8)
        self.place_at_grid(lbl2, "C6", scale_factor=0.8)
        
        add_eq = MathTex(r"v_1 + v_2 = v_{sum}", color=COLOR_MATH)
        # Fix Issue 18: Reposition add_eq
        self.place_in_area(add_eq, 'B3', 'B5', scale_factor=1.0)
        
        self.play(Create(vec1), FadeIn(lbl1))
        self.wait(0.5)
        self.play(Create(vec2), FadeIn(lbl2))
        self.wait(0.5)
        self.play(Create(vec_s), Write(add_eq))
        
        self.wait(3)
        
        # Final cleanup
        self.play(
            *[FadeOut(m) for m in [vec1, vec2, vec_s, lbl1, lbl2, add_eq]],
            self.lecture[2].animate.set_color(WHITE)
        )
        self.wait(1)
