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
        lecture_lines = [
            "Calculate using 3x3 matrix determinant.",
            "Organize components using unit vectors.",
            "Expand along the top row.",
            "Algorithm provides precise vector results.",
            "Practice with specific vector examples."
        ]
        self.setup_layout("Computational Method: The Determinant", lecture_lines)
        
        # Matrix components
        matrix_str = r"\begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ a_x & a_y & a_z \\ b_x & b_y & b_z \end{vmatrix}"
        matrix = MathTex(matrix_str, font_size=36)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_at_grid(matrix, 'B3', scale_factor=0.9)
        self.play(Write(matrix))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        # Highlighting row 1 (i, j, k)
        row1_highlight = matrix[0][0:9]
        self.play(row1_highlight.animate.set_color("#FFFF00"))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        expansion = MathTex(r"= \mathbf{i}(a_y b_z - a_z b_y) - \mathbf{j}(a_x b_z - a_z b_x) + \mathbf{k}(a_x b_y - a_y b_x)", font_size=28)
        self.place_in_area(expansion, 'C4', 'D6', scale_factor=0.75)
        self.play(Write(expansion))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF0000"))
        final_vec = MathTex(r"\mathbf{v} = \langle v_x, v_y, v_z \rangle", font_size=32, color="#FF0000")
        self.place_at_grid(final_vec, 'F3', scale_factor=0.9)
        self.play(FadeIn(final_vec))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        calculator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        self.place_at_grid(calculator, 'F6', scale_factor=0.5)
        
        example = MathTex(r"\mathbf{A} \times \mathbf{B} = \langle 1, 2, 3 \rangle \times \langle 4, 5, 6 \rangle", font_size=24)
        self.place_at_grid(example, 'F6', scale_factor=0.8)
        # Move calculator off to avoid total overlap
        calculator.shift(LEFT * 1.5)
        
        self.play(FadeIn(calculator), Write(example))
        self.wait(2)
