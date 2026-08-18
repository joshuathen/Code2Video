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
        self.setup_layout("From Transformation to Matrix", [
            "Where basis vectors land defines the transformation matrix.",
            "The coordinates become columns of the new matrix.",
            "This maps geometry directly into algebraic form."
        ])
        
        # Matrix Setup
        matrix = Matrix([[2, 0], [0, 2]])
        self.place_in_area(matrix, 'B3', 'C5', scale_factor=1.1)
        
        # Asset for animation
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        self.place_at_grid(icon, 'A4', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Create(matrix))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        # Highlighting the columns
        column1_rect = SurroundingRectangle(matrix.get_columns()[0], color=YELLOW)
        column2_rect = SurroundingRectangle(matrix.get_columns()[1], color=YELLOW)
        
        # Show transformation with asset
        self.play(FadeIn(icon))
        self.play(Create(column1_rect))
        self.wait(0.5)
        self.play(ReplacementTransform(column1_rect, column2_rect))
        self.wait(0.5)
        self.play(FadeOut(column2_rect), FadeOut(icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        # Algebraic mapping text
        map_text = MathTex(r"T\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}")
        self.place_at_grid(map_text, 'E4', scale_factor=0.8)
        self.play(Write(map_text))
        self.wait(2)
