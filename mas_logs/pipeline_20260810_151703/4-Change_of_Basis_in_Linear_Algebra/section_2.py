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
        self.setup_layout("Prerequisite Review: Basis as Building Blocks", 
                         ["Basis vectors are simple building blocks.", 
                          "Standard grid lines become skewed.", 
                          "Skewed grids represent new basis vectors."])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF4500")
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg
        grid_bg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_in_area(grid_bg, "A1", "F6", scale_factor=0.6)
        self.add(grid_bg)
        self.bring_to_back(grid_bg) # Background behind vectors

        # Create standard basis
        basis_i = Arrow(start=ORIGIN, end=RIGHT, color="#FF4500")
        basis_j = Arrow(start=ORIGIN, end=UP, color="#FF4500")
        # Fixed per Issue 25
        self.place_at_grid(basis_i, "D4", scale_factor=1.0)
        self.place_at_grid(basis_j, "C4", scale_factor=1.0)
        self.play(Create(basis_i), Create(basis_j))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#1E90FF")
        
        # Transform into skewed
        skewed_u = Arrow(start=ORIGIN, end=2*RIGHT + UP, color="#1E90FF")
        skewed_v = Arrow(start=ORIGIN, end=-RIGHT + UP, color="#1E90FF")
        
        # Using a VGroup to track original vectors during transform for better stability
        vgroup = VGroup(basis_i, basis_j)
        self.play(
            Transform(basis_i, skewed_u),
            Transform(basis_j, skewed_v)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFFFF")
        
        # Fixed per Issue 26
        origin_label = Tex(r"O", color=WHITE)
        self.place_at_grid(origin_label, "D3", scale_factor=0.7)
        
        # Fixed per Issue 24
        label_u = MathTex(r"\\vec{u}", color=WHITE)
        label_v = MathTex(r"\\vec{v}", color=WHITE)
        
        self.place_at_grid(label_u, "D5", scale_factor=0.8)
        self.place_at_grid(label_v, "C3", scale_factor=0.8)
        
        self.play(Write(origin_label), Write(label_u), Write(label_v))
        self.wait(2)
