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

class Section5Scene(TeachingScene):
    def construct(self):
        # 1. Setup Layout
        title = "The Mathematical Transformation"
        lecture_lines = [
            "We translate new coordinates using matrix multiplication.",
            "Matrix P converts pirate coordinates to standard ones.",
            "P-inverse performs the reverse translation back to the pirate.",
            "The formula [v]_standard equals P times [v]_new.",
            "This math keeps everyone's perspective perfectly aligned."
        ]
        self.setup_layout(title, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Display formula [v]_std = P * [v]_new in white (#FFFFFF)
        # Using Belief B002: center formula using an area span.
        formula = MathTex(
            r"[\vec{v}]_{standard} = P \cdot [\vec{v}]_{new}",
            color="#FFFFFF", font_size=36
        )
        self.place_in_area(formula, "B2", "B5", scale_factor=1.0)
        
        self.play(Write(formula))
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show the Pirate icon and Pirate's vector [1, 1] in yellow (#FFFF00) next to the formula.
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pirate.svg
        pirate_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pirate.svg")
        pirate_icon.set_color("#FFFF00")
        self.place_at_grid(pirate_icon, "B1", scale_factor=0.6) # Shifted to B1 to avoid overlap with formula
        
        v_new_label = MathTex(
            r"[\vec{v}]_{new} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}",
            color="#FFFF00", font_size=32
        )
        self.place_at_grid(v_new_label, "B6", scale_factor=0.9)

        self.play(FadeIn(pirate_icon), Write(v_new_label))
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Matrix P converts pirate coordinates to standard ones.
        # P-inverse performs the reverse translation.
        # Let's show Matrix P explicitly for context.
        matrix_p_label = MathTex(
            r"P = \begin{bmatrix} 2 & -1 \\ 1 & 1 \end{bmatrix}",
            color="#FFFFFF", font_size=32
        )
        self.place_at_grid(matrix_p_label, "C2", scale_factor=0.9)
        
        self.play(Write(matrix_p_label))
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Substitute matrix P and vector [1, 1] into the matrix multiplication.
        # Calculation: [v]_std = P * [v]_new
        substitution = MathTex(
            r"\begin{bmatrix} 2 & -1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}",
            color="#FFFFFF", font_size=32
        )
        self.place_in_area(substitution, "C3", "C6", scale_factor=1.0)
        
        self.play(Write(substitution))
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight the resulting vector [1, 2] on the standard grid in white (#FFFFFF).
        origin_ref = "E4"
        grid = NumberPlane(
            x_range=[-1, 3, 1],
            y_range=[-1, 3, 1],
            x_length=2.5,
            y_length=2.5,
            axis_config={"include_tip": True, "stroke_width": 2},
            background_line_style={"stroke_opacity": 0.3}
        )
        self.place_at_grid(grid, origin_ref, scale_factor=1.0)
        
        v_std_vec = Arrow(
            grid.get_origin(), 
            grid.c2p(1, 2), 
            buff=0, 
            color="#FFFFFF", 
            stroke_width=4
        )
        # Using Belief B012: place label near the mobject.
        v_std_coord = MathTex(
            r"[\vec{v}]_{standard} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}", 
            color="#FFFFFF", 
            font_size=28
        )
        self.place_at_grid(v_std_coord, "D5", scale_factor=1.0)

        self.play(Create(grid), GrowArrow(v_std_vec), FadeIn(v_std_coord))
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        self.wait(3)
