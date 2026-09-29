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
            "We represent this operation using a 3x3 matrix.",
            "Place basis vectors i, j, k at the top.",
            "Determinant expansion computes the resulting vector components.",
            "Remember the anti-commutative property: A x B = -B x A.",
            "Example: i x j gives the unit vector k."
        ]
        self.setup_layout("The Determinant Formula for 3D", lecture_lines)
        
        # 3x3 Matrix for Cross Product
        matrix = MathTex(
            "\\begin{pmatrix} \\mathbf{i} & \\mathbf{j} & \\mathbf{k} \\\\ a_1 & a_2 & a_3 \\\\ b_1 & b_2 & b_3 \\end{pmatrix}",
            font_size=40
        )
        self.place_at_grid(matrix, 'B4', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.play(Write(matrix))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FFFF")
        i_box = SurroundingRectangle(matrix[0][0:3], color="#FF00FF", buff=0.1)
        
        # Asset usage
        asset_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        self.place_in_area(asset_icon, 'C1', 'C2', scale_factor=0.3)
        
        self.play(Create(i_box), FadeIn(asset_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FFFF")
        formula = MathTex("\\mathbf{i}(a_2b_3 - a_3b_2) - \\mathbf{j}(a_1b_3 - a_3b_1) + \\mathbf{k}(a_1b_2 - a_2b_1)", font_size=32)
        self.place_in_area(formula, 'D2', 'E5', scale_factor=0.8)
        self.play(ReplacementTransform(matrix.copy(), formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FFFF00")
        prop = MathTex("\\mathbf{a} \\times \\mathbf{b} = -(\\mathbf{b} \\times \\mathbf{a})", font_size=36, color="#FFFF00")
        self.place_at_grid(prop, 'C5', scale_factor=0.7)
        self.play(FadeIn(prop))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#00FF00")
        ex = MathTex("\\mathbf{i} \\times \\mathbf{j} = \\mathbf{k}", font_size=36, color="#00FF00")
        self.place_at_grid(ex, 'F3', scale_factor=1.0)
        self.play(FadeIn(ex))
        self.wait(2)
