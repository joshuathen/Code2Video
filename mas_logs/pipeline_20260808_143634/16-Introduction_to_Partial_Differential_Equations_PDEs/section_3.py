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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "The heat equation links time and space.",
            "The Laplacian represents neighbor temperature differences.",
            "Heat diffuses from dense to empty regions."
        ]
        self.setup_layout("The Heat Equation: A Practical Foundation", lecture_lines)
        
        # Load assets
        rod = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg", color=WHITE)
        formula = MathTex(r"\frac{\partial u}{\partial t} = \alpha \nabla^2 u", font_size=36, color=GREEN)
        
        # Progress bar (the rod icon repurposed as a flow indicator)
        progress_bar = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg", color=YELLOW)

        # === Animation for Lecture Line 1 ===
        # Fix 29: Anchor title (already anchored by setup_layout but we ensure it works)
        # self.place_at_grid(self.title, 'A1', scale_factor=1.0)
        self.place_at_grid(rod, 'C3', scale_factor=2.0)
        self.play(FadeIn(rod))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Fix 27: Position formula
        self.place_in_area(formula, 'B4', 'C6', scale_factor=1.2)
        self.lecture[1].set_color(YELLOW)
        self.play(FadeIn(formula))
        # rod color change
        rod.set_color(RED)
        self.play(rod.animate.set_color_by_gradient(BLUE, RED))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fix 28: Progress bar
        self.place_in_area(progress_bar, 'E2', 'E5', scale_factor=1.0)
        self.lecture[2].set_color(YELLOW)
        self.play(FadeIn(progress_bar))
        self.play(Indicate(progress_bar))
        self.wait(2)
