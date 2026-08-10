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
        lecture_lines = ["ODEs track change in one variable.", "PDEs track change in multiple variables.", "Think of a moving point.", "Now think of a rippling surface.", "PDEs capture this multi-variable complexity."]
        self.setup_layout("From ODEs to PDEs: A Visual Shift", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        line_ode = Line(start=self.grid["B2"], end=self.grid["B5"], color=WHITE)
        self.play(Create(line_ode))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 2 ===
        # Using place_in_area as suggested by Critic
        grid_pde = NumberPlane(
            x_range=(-2, 2), y_range=(-1, 1),
            x_length=3, y_length=1.5,
            background_line_style={"stroke_color": "#00FF00", "stroke_width": 2}
        )
        self.place_in_area(grid_pde, "C1", "D3", scale_factor=0.7)
        self.play(FadeIn(grid_pde))
        self.play(self.lecture[1].animate.set_color("#00FF00"))

        # === Animation for Lecture Line 3 ===
        # Use Asset: particle.svg
        particle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg", color=YELLOW)
        self.place_at_grid(particle, "B3", scale_factor=0.3)
        self.play(FadeIn(particle))
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # === Animation for Lecture Line 4 ===
        # Rippling surface using place_at_grid as suggested by Critic
        surface = Surface(
            lambda u, v: np.array([u, v, 0.2 * np.sin(2 * (u**2 + v**2))]),
            u_range=[-1, 1], v_range=[-1, 1]
        ).set_color(BLUE)
        self.place_at_grid(surface, "E4", scale_factor=0.5)
        self.play(FadeIn(surface))
        self.play(self.lecture[3].animate.set_color("#0000FF"))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF00FF"))
        self.wait(2)
