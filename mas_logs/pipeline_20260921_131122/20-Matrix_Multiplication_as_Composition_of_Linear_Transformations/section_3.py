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
        lecture_lines = ["Composition is written as A(Bv).", "This equals the matrix product AB.", "AB captures the combined transformation instruction."]
        self.setup_layout("Deriving the Matrix Product", lecture_lines)
        
        matrix_a = Matrix([[ "a_{11}", "a_{12}" ], [ "a_{21}", "a_{22}" ]], h_buff=1.0, v_buff=0.8).scale(0.5)
        matrix_b = Matrix([[ "b_{11}", "b_{12}" ], [ "b_{21}", "b_{22}" ]], h_buff=1.0, v_buff=0.8).scale(0.5)
        matrix_ab = Matrix([[ "c_{11}", "c_{12}" ], [ "c_{21}", "c_{22}" ]], h_buff=1.0, v_buff=0.8).scale(0.5)
        
        # Load asset
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        # In a real scenario, this would be an SVGMobject, but it's an empty icon.
        # We can treat it as a placeholder.
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg").scale(0.1)

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(matrix_a, "B2", scale_factor=0.6)
        self.place_at_grid(matrix_b, "B3", scale_factor=0.6)
        self.play(FadeIn(matrix_a), FadeIn(matrix_b))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(GRAY), self.lecture[1].animate.set_color("#FFFFFF"))
        self.place_at_grid(matrix_ab, "D3", scale_factor=0.6)
        self.play(FadeIn(matrix_ab))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(GRAY), self.lecture[2].animate.set_color("#FFFFFF"))
        
        # Highlight columns
        matrix_ab.get_columns()[0].set_color("#FF4500")
        matrix_ab.get_columns()[1].set_color("#00CED1")
        
        # Use the asset (just adding it near the matrix)
        icon.next_to(matrix_ab, UP)
        self.play(FadeIn(icon), Indicate(matrix_ab))
        self.wait(2)
