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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Defining the Dot Product as a Linear Map", ["Dot product is a linear map.", "It returns a scalar from two vectors.", "A fixed vector w defines this function."])
        
        # Elements
        w_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", color="#FF0000")
        matrix_m = MathTex(r"M = \begin{bmatrix} w_1 & w_2 \end{bmatrix}", color=WHITE)
        vector_v = MathTex(r"v = \begin{bmatrix} x \\ y \end{bmatrix}", color="#FF0000")
        result = MathTex(r"f(v) = w \cdot v = x \cdot w_1 + y \cdot w_2", color="#00FF00")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.place_at_grid(w_icon, "A3", scale_factor=0.5)
        self.play(FadeIn(w_icon))
        self.place_in_area(matrix_m, 'B2', 'B5', scale_factor=0.9)
        self.play(Write(matrix_m))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        self.place_at_grid(vector_v, 'C3', scale_factor=0.9)
        self.play(Write(vector_v))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.place_in_area(result, 'D2', 'E5', scale_factor=1.0)
        self.play(Write(result))
        self.wait(1)
