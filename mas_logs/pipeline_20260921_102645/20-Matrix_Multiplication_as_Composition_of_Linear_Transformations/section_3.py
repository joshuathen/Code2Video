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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Composition A(B(v)) equals one matrix C.",
            "Matrix product C equals A times B.",
            "Observe basis paths through B then A.",
            "This determines the product's column values.",
            "Order follows column-by-column logic strictly."
        ]
        self.setup_layout("Deriving the Matrix Product", lecture_lines)
        
        # Pre-load SVG asset
        vector_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vector.svg")

        # === Animation for Lecture Line 1 ===
        # Composition A(B(v)) equals one matrix C.
        self.lecture[0].set_color(BLUE)
        formula = MathTex(r"A(B(v)) = C(v)", font_size=36)
        self.place_at_grid(formula, "B2")
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Matrix product C equals A times B.
        self.lecture[1].set_color(YELLOW)
        product_eq = MathTex(r"C = AB", font_size=48, color=YELLOW)
        self.place_at_grid(product_eq, "C2")
        
        # Using asset
        self.place_at_grid(vector_icon, "C5", scale_factor=0.5)
        self.play(FadeIn(product_eq), FadeIn(vector_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Observe basis paths through B then A.
        self.lecture[2].set_color(GREEN)
        basis = MathTex(r"\hat{i}, \hat{j} \xrightarrow{B} \dots \xrightarrow{A} \dots", font_size=32)
        self.place_at_grid(basis, "D2")
        self.play(Write(basis))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This determines the product's column values.
        self.lecture[3].set_color(RED)
        cols_text = Text("Columns of C", font_size=28, color=RED)
        self.place_at_grid(cols_text, "E2")
        self.play(Write(cols_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Order follows column-by-column logic strictly.
        self.lecture[4].set_color(PURPLE)
        order_text = Text("Column-by-Column", font_size=28, color=PURPLE)
        self.place_at_grid(order_text, "F2")
        self.play(Write(order_text))
        self.wait(2)
