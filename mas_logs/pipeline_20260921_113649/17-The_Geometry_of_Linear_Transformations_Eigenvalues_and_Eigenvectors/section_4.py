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
        self.setup_layout("Constructing an Eigenbasis", [
            "Eigenvectors can form a new coordinate basis.",
            "This is called an eigenbasis for the transformation.",
            "Matrices become diagonal in an eigenbasis."
        ])
        
        # Assets
        grid_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_in_area(grid_icon, "A1", "F6", scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        basis_label = Text("Basis", color=WHITE)
        self.place_at_grid(basis_label, "A4", scale_factor=0.8)
        self.play(FadeIn(grid_icon), Write(basis_label), self.lecture[0].animate.set_color(WHITE))
        
        # === Animation for Lecture Line 2 ===
        v1 = Vector([1, 1], color=GREEN)
        v2 = Vector([-1, 2], color=GREEN)
        self.place_at_grid(v1, "C3", scale_factor=0.9)
        self.place_at_grid(v2, "C4", scale_factor=0.9)
        eigen_label = Text("Eigenbasis", color="#00FF00")
        self.place_at_grid(eigen_label, "B4", scale_factor=0.7)
        self.play(Create(v1), Create(v2), Write(eigen_label), self.lecture[1].animate.set_color("#00FF00"))
        
        # === Animation for Lecture Line 3 ===
        diag_label = Text("Diagonal", color="#FFFF00")
        self.place_at_grid(diag_label, "B5", scale_factor=0.7)
        self.play(grid_icon.animate.set_color("#FFFF00"), Write(diag_label), self.lecture[2].animate.set_color("#FFFF00"))
