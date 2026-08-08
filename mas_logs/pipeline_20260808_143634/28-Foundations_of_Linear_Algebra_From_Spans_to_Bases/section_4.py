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
        self.setup_layout("Bases: The Minimum Building Blocks", [
            "Bases are minimal spanning sets.", 
            "They must be linearly independent.", 
            "They form a coordinate system."
        ])
        
        # Assets
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        vector_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vector.svg")
        
        # Grid setup (constrained area)
        self.place_in_area(grid_asset, 'B2', 'E5', scale_factor=0.5)
        
        # Vector objects
        i_vec = Arrow(start=ORIGIN, end=RIGHT, color=BLUE)
        j_vec = Arrow(start=ORIGIN, end=UP, color=RED)
        vectors = VGroup(i_vec, j_vec)
        self.place_in_area(vectors, 'C2', 'D4', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(grid_asset), Create(vectors))
        self.play(self.lecture[0].animate.set_color(BLUE))

        # === Animation for Lecture Line 2 ===
        k_vec = Arrow(start=ORIGIN, end=RIGHT+UP, color=YELLOW)
        self.place_in_area(k_vec, 'C2', 'D4', scale_factor=0.6)
        
        self.play(Create(k_vec))
        self.play(FadeOut(k_vec)) 
        self.play(self.lecture[1].animate.set_color(RED))

        # === Animation for Lecture Line 3 ===
        basis_label = Text("Basis {i, j}", font_size=24, color=WHITE)
        self.place_at_grid(basis_label, 'B1', scale_factor=0.7)
        
        self.play(Write(basis_label))
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.wait(2)
