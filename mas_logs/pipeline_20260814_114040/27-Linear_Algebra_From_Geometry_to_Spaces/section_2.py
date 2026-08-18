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
        self.setup_layout("Inverse Matrices: The 'Undo' Button", [
            "Inverse matrices reverse linear transformations.",
            "They act like undo buttons for space.",
            "Applying A then inverse restores original points."
        ])

        # Assets
        btn_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/button.svg"
        
        # Animation Setup
        vector_orig = Vector([1, 0.5], color="#FF5733")
        matrix_a = MathTex(r"A = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}", color="#FFFFFF")
        matrix_inv = MathTex(r"A^{-1} = \begin{pmatrix} 0.5 & 0 \\ 0 & 0.5 \end{pmatrix}", color="#33FF57")
        
        button_a = SVGMobject(btn_asset)
        button_inv = SVGMobject(btn_asset)
        
        # Apply mandatory layout fixes (VideoCritic requests)
        self.place_in_area(matrix_a, 'A2', 'C3', scale_factor=0.7)
        self.place_at_grid(vector_orig, 'D2', scale_factor=0.6)
        self.place_in_area(button_a, 'A1', 'C1', scale_factor=0.5)
        
        vector_trans = vector_orig.copy().scale(2)
        vector_trans.set_color("#FF5733")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(FadeIn(matrix_a), FadeIn(button_a), FadeIn(vector_orig))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#33FF57")
        self.place_in_area(matrix_inv, 'A4', 'C5', scale_factor=0.7)
        self.place_in_area(button_inv, 'A6', 'C6', scale_factor=0.5)
        self.play(FadeIn(matrix_inv), FadeIn(button_inv))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF5733")
        vector_final = vector_orig.copy()
        
        # Visualize restoration
        self.play(Transform(vector_orig, vector_trans))
        self.wait(0.5)
        self.play(Transform(vector_orig, vector_final), run_time=1.5)
        self.wait(1)
