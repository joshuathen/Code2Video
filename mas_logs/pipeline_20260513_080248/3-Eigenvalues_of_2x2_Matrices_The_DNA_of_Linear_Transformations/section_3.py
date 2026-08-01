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
        # Setup title and lecture lines
        title_str = "The Characteristic Equation (The Math Tool)"
        lines = [
            "Rearrange the eigenvalue equation to equal zero.",
            "We seek a non-zero solution for vector v.",
            "This requires the transformation matrix to be singular.",
            "Thus, the determinant of the matrix must be zero.",
            "Subtract lambda from the diagonal elements to solve."
        ]
        self.setup_layout(title_str, lines)

        # === Animation for Lecture Line 1 ===
        # Transition: 'A v = λ v' to '(A - λ I) v = 0'
        self.lecture[0].set_color(WHITE)
        
        eq1 = VGroup(
            Text("A", color=WHITE),
            Text("v", color=WHITE),
            Text(" = ", color=WHITE),
            Text("\u03BB", color=WHITE),
            Text("v", color=WHITE)
        ).arrange(RIGHT, buff=0.1)
        self.place_in_area(eq1, "A2", "B5", scale_factor=1.2)
        
        self.play(Write(eq1))
        self.wait(1)
        
        eq2 = VGroup(
            Text("(", color=WHITE),
            Text("A", color=WHITE),
            Text(" - ", color=WHITE),
            Text("\u03BB", color=WHITE),
            Text("I", color=WHITE),
            Text(")", color=WHITE),
            Text("v", color=WHITE),
            Text(" = ", color=WHITE),
            Text("0", color=WHITE)
        ).arrange(RIGHT, buff=0.1)
        self.place_in_area(eq2, "A2", "B5", scale_factor=1.2)
        
        self.play(ReplacementTransform(eq1, eq2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(WHITE)
        # Condition: non-zero solution for vector v
        v_neq_zero = Text("v \u2260 0", color=WHITE, font_size=32)
        self.place_at_grid(v_neq_zero, "B6", scale_factor=1.0)
        self.play(Write(v_neq_zero))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        # Red box around (A - λ I) and label 'Singular Matrix'
        red_box = SurroundingRectangle(eq2[:6], color="#FF0000", buff=0.1)
        
        singular_text = Text("Singular Matrix", font_size=24, color=WHITE)
        # Resolved Issue 33: Fixed placement to avoid crowding
        self.place_in_area(singular_text, "C2", "C6", scale_factor=0.8)
        
        self.play(Create(red_box), FadeIn(singular_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        
        # 'det(A - λ I) = 0' appearing in yellow (#FFFF00)
        det_eq = VGroup(
            Text("det", color=YELLOW),
            Text("(", color=YELLOW),
            Text("A", color=YELLOW),
            Text(" - ", color=YELLOW),
            Text("\u03BB", color=YELLOW),
            Text("I", color=YELLOW),
            Text(")", color=YELLOW),
            Text(" = ", color=YELLOW),
            Text("0", color=YELLOW)
        ).arrange(RIGHT, buff=0.1)
        # Resolved Issue 34: Rescaled and adjusted area for margin
        self.place_in_area(det_eq, "D2", "D5", scale_factor=1.0)
        
        self.play(Write(det_eq))
        # Glow emphasis
        self.play(Indicate(det_eq, color=YELLOW, scale_factor=1.1))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFD700") # Gold
        
        # 2x2 matrix with subtraction on diagonal
        b_left = Text("[", font_size=80, color="#FFD700")
        b_right = Text("]", font_size=80, color="#FFD700")
        
        a_minus_lam = Text("a - \u03BB", color="#FFD700", font_size=32)
        b_elem = Text("b", color="#FFD700", font_size=32)
        c_elem = Text("c", color="#FFD700", font_size=32)
        d_minus_lam = Text("d - \u03BB", color="#FFD700", font_size=32)
        
        row1 = VGroup(a_minus_lam, b_elem).arrange(RIGHT, buff=1.2)
        row2 = VGroup(c_elem, d_minus_lam).arrange(RIGHT, buff=1.2)
        
        a_minus_lam.align_to(c_elem, LEFT)
        b_elem.align_to(d_minus_lam, LEFT)
        
        elements = VGroup(row1, row2).arrange(DOWN, buff=0.6)
        matrix_vgroup = VGroup(b_left, elements, b_right).arrange(RIGHT, buff=0.2)
        
        # Resolved Issue 35: Adjusted area for horizontal padding
        self.place_in_area(matrix_vgroup, "E2", "F6", scale_factor=0.9)
        
        self.play(FadeIn(matrix_vgroup))
        self.wait(2)
