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
        self.setup_layout("The Algebraic Formula (The Determinant)", [
            "We use a 3x3 determinant to compute it.", 
            "Basis vectors i, j, k define the frame.", 
            "Cross-multiplication calculates the x, y, z components."
        ])
        
        # Matrix Setup
        matrix_str = r"""
        \begin{pmatrix}
        \mathbf{i} & \mathbf{j} & \mathbf{k} \\
        a_x & a_y & a_z \\
        b_x & b_y & b_z
        \end{pmatrix}
        """
        matrix = MathTex(matrix_str, font_size=40)
        # Applying requested layout fixes: (35/37) A3, C6, 0.85
        self.place_in_area(matrix, "A3", "C6", scale_factor=0.85)

        # Assets - Using SVG placeholder icons as required
        asset_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        asset_icon.scale(0.5).next_to(matrix, RIGHT)

        # === Animation for Lecture Line 1 ===
        self.play(Write(matrix), FadeIn(asset_icon))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        row1_highlight = matrix[0][0:3]
        self.play(
            row1_highlight.animate.set_color("#FF6666"),
            self.lecture[1].animate.set_color("#FF6666")
        )

        # === Animation for Lecture Line 3 ===
        det_text = MathTex(
            r"\text{Det} = \mathbf{i}(a_y b_z - a_z b_y) - \mathbf{j}(a_x b_z - a_z b_x) + \mathbf{k}(a_x b_y - a_y b_x)",
            font_size=28
        )
        # Applying requested layout fix (24/36) E2, 0.9
        self.place_at_grid(det_text, "E2", scale_factor=0.9)
        
        self.play(
            FadeIn(det_text),
            self.lecture[2].animate.set_color("#66CCFF")
        )
        self.wait(2)
