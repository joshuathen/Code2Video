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
        lecture_lines = ["Basis is the minimal set to span.", "It must be linearly independent.", "Basis is the most efficient instruction manual."]
        self.setup_layout("The Efficiency Expert: Bases", lecture_lines)
        
        # Elements
        v1 = Vector([1, 0.5], color=BLUE)
        v2 = Vector([-0.5, 1], color=RED)
        basis_group = VGroup(v1, v2).arrange(RIGHT)
        
        # Asset
        manual_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/manual.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        basis_text = Text("Basis = Minimal Spanning Set", font_size=24)
        self.place_at_grid(basis_text, 'B3', scale_factor=0.85)
        self.place_at_grid(manual_icon, 'A3', scale_factor=0.3)
        self.play(FadeIn(basis_text), FadeIn(manual_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(RED)
        self.place_in_area(basis_group, 'C2', 'D5', scale_factor=0.9)
        self.play(FadeIn(basis_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        efficient_text = Text("Efficiency: Optimal Description", font_size=24)
        self.place_at_grid(efficient_text, 'E4', scale_factor=0.8)
        self.play(Write(efficient_text))
        self.wait(2)
