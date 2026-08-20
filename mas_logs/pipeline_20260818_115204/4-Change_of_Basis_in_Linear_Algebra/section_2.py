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
        self.setup_layout("Prerequisite Recap: The Matrix as a Translator", [
            "A basis spans the vector space.",
            "Transformation matrices map basis vectors.",
            "Standard basis i, j land on b1, b2."
        ])
        
        # Asset path
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg"
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        grid_asset = SVGMobject(asset_path).set_color("#FFFFFF")
        self.place_at_grid(grid_asset, "B2", scale_factor=0.5)
        
        matrix_m = MathTex(r"M = \begin{pmatrix} a & b \\ c & d \end{pmatrix}", color="#FFFFFF")
        self.place_at_grid(matrix_m, "B2", scale_factor=0.8)
        self.play(FadeIn(grid_asset), Write(matrix_m))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF00FF")
        
        # Grid transformation asset
        grid_trans = SVGMobject(asset_path).set_color("#FF00FF")
        self.place_at_grid(grid_trans, "C4", scale_factor=0.6)
        
        vec_x = Vector([1, 0], color="#00FF00")
        self.place_in_area(vec_x, 'D2', 'D4', scale_factor=0.9)
        
        y_label = MathTex(r"Y = MX", color="#00FF00")
        self.place_at_grid(y_label, 'D5', scale_factor=0.7)
        
        self.play(FadeIn(grid_trans), GrowArrow(vec_x), Write(y_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FFFF")
        
        m_inv = MathTex(r"M^{-1}", color="#FF0000")
        self.place_at_grid(m_inv, 'C2', scale_factor=0.9)
        
        # Show mapping back
        self.play(FadeIn(m_inv), vec_x.animate.set_color("#FF0000"), run_time=1.5)
        self.wait(2)
