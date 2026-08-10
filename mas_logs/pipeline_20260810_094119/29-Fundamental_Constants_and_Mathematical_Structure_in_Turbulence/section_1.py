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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Turbulence is a hierarchical structure of energy cascades.",
            "It is not random noise, but ordered patterns.",
            "Energy moves from large scales to small scales.",
            "Reynolds number describes inertial versus viscous forces.",
            "High Reynolds numbers lead to chaotic, dissipating vortices."
        ]
        self.setup_layout("The Turbulence Puzzle: Introduction", lecture_lines)
        
        # Load Assets
        vortex_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vortex.svg")
        fluid_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fluid.svg")

        # Visual elements
        dots = VGroup(*[Dot(radius=0.05, color=BLUE) for _ in range(40)])
        dots_and_labels = VGroup(dots)
        
        self.place_in_area(dots_and_labels, 'B4', 'E6', scale_factor=0.35)
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(vortex_icon, 'C2', scale_factor=0.5)
        self.play(FadeIn(self.title), FadeIn(vortex_icon))
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Create(dots))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.play(FadeOut(vortex_icon), FadeIn(fluid_icon.move_to(self.grid['C3']).scale(0.5)))
        self.play(dots.animate.set_color(TEAL))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(dots.animate.scale(0.8))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(ORANGE))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF4500"))
        self.play(dots.animate.set_color("#FF4500").set_opacity(0.5))
        self.play(FadeOut(self.title), FadeOut(fluid_icon))
        
        self.wait(1)
