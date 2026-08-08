from manim import *

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
        # Define the content
        title = "Abstract Example: The Space of Polynomials"
        lecture_lines = [
            "Consider the set of all low-degree polynomials.",
            "Adding two polynomials results in another polynomial.",
            "We can even visualize coefficients as coordinates."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # Colors
        GOLD = "#FFD700"
        CYAN = "#00FFFF"
        ORANGE = "#FFA500"
        WHITE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Consider the set of all low-degree polynomials.
        self.lecture[0].set_color(GOLD)
        
        poly_p = MathTex("P(x) = ax^2 + bx + c", color=GOLD)
        # Issue 36 Fix: Move poly_p to A1-A4 to prevent horizontal overlap
        self.place_in_area(poly_p, "A1", "A4", scale_factor=0.9)
        
        self.play(Write(poly_p))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Adding two polynomials results in another polynomial.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(WHITE_COLOR)
        
        poly_q = MathTex("Q(x) = dx^2 + ex + f", color=WHITE_COLOR)
        # Issue 36 Fix: Move poly_q to B1-B4 to prevent horizontal overlap
        self.place_in_area(poly_q, "B1", "B4", scale_factor=0.9)
        
        poly_sum = MathTex(
            "P(x) + Q(x) = (a+d)x^2 + (b+e)x + (c+f)", 
            color=WHITE_COLOR
        )
        # Issue 37 Fix: Move poly_sum to C1-C6 (single row)
        self.place_in_area(poly_sum, "C1", "C6", scale_factor=0.8)
        
        self.play(Write(poly_q))
        self.wait(1)
        self.play(Write(poly_sum))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # We can even visualize coefficients as coordinates.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(CYAN)
        
        # Scalar multiplication k*P(x)
        scalar_k = MathTex("k \\cdot P(x) = (ka)x^2 + (kb)x + (kc)", color=CYAN)
        # Issue 37 Fix: Move scalar_k to D1-D6
        self.place_in_area(scalar_k, "D1", "D6", scale_factor=0.8)
        
        # Transform P(x) to show coordinate view
        vector_p = MathTex(r"\text{Coordinates: } \begin{bmatrix} a \\ b \\ c \end{bmatrix}", color=GOLD)
        # Issue 35 Fix: Move vector_p to A5-B6 to avoid overlap with poly_q
        self.place_in_area(vector_p, "A5", "B6", scale_factor=0.8)
        
        # "Polynomial Space" label
        label = Text("Polynomial Space", color=ORANGE, font_size=24)
        self.place_in_area(label, "F1", "F6", scale_factor=1.0)
        
        self.play(
            poly_p.animate.set_color(CYAN),
            FadeOut(poly_q),
            FadeOut(poly_sum),
            Write(scalar_k)
        )
        self.wait(1)
        
        self.play(
            ReplacementTransform(poly_p.copy(), vector_p),
            Write(label)
        )
        self.wait(3)
