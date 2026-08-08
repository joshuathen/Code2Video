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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Divergence depends on our metric.", "Real lines explode to infinity.", "2-adic portals capture them."]
        self.setup_layout("Convergence vs. Divergence: The Shift", lecture_lines)
        
        # Elements
        real_line = NumberLine(x_range=[0, 10, 1], length=5).set_color(WHITE)
        adic_line = NumberLine(x_range=[-2, 2, 1], length=5).set_color(WHITE)
        
        # Asset
        portal = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/portal.svg")
        
        # Labels
        real_label = Text("Real Line", font_size=20)
        adic_label = Text("2-adic Tree", font_size=20)
        
        # Positioning (Applying Fixes from Issues 29, 30, 31)
        self.place_at_grid(real_line, 'B3', scale_factor=0.7)
        self.place_at_grid(adic_line, 'E3', scale_factor=0.7)
        self.place_at_grid(real_label, 'B2', scale_factor=0.6)
        self.place_at_grid(adic_label, 'E2', scale_factor=0.6)
        
        # Divergent sequence (Real)
        real_dots = VGroup(*[Dot(real_line.n2p(2**i), color=RED) for i in range(4)])
        
        # Convergent sequence (2-adic)
        adic_dots = VGroup(*[Dot(adic_line.n2p(-1 + 1/(2**i)), color=GREEN) for i in range(4)])

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(real_line), FadeIn(adic_line), FadeIn(real_label), FadeIn(adic_label))
        self.place_at_grid(portal, 'D3', scale_factor=0.2)
        self.play(FadeIn(portal))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        self.play(Create(real_dots))
        self.play(real_dots.animate.shift(RIGHT * 1))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        self.play(Create(adic_dots))
        self.play(portal.animate.set_color("#00FF00"))
        self.wait(1)
