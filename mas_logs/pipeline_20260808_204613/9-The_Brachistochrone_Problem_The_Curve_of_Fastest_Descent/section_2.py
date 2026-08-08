from manim import *
import numpy as np

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
        lecture_lines = [
            "Energy conservation governs the bead's speed.",
            "Potential energy converts into kinetic energy.",
            "Velocity depends on the vertical drop height."
        ]
        self.setup_layout("Prerequisite Concept: Conservation of Energy", lecture_lines)
        
        # Initialize formula mobjects
        u_formula = MathTex(r"U = mgh", color=WHITE)
        k_formula = MathTex(r"K = \frac{1}{2} mv^2", color=YELLOW)
        total_formula = MathTex(r"U + K = \text{constant}", color=GREEN)
        
        # Initialize assets
        bead_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/bead.svg"
        bead1 = SVGMobject(bead_asset_path)
        bead2 = SVGMobject(bead_asset_path)
        
        # Position them
        self.place_at_grid(u_formula, 'B3', scale_factor=0.8)
        self.place_at_grid(k_formula, 'D3', scale_factor=0.8)
        self.place_at_grid(total_formula, 'F3', scale_factor=0.8)
        
        # Position assets
        self.place_at_grid(bead1, 'B5', scale_factor=0.4)
        self.place_at_grid(bead2, 'F5', scale_factor=0.4)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(u_formula), FadeIn(bead1))
        self.lecture[0].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(k_formula))
        self.lecture[1].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(total_formula), FadeIn(bead2))
        self.lecture[2].set_color(GREEN)
        self.wait(1)
