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
        self.setup_layout("Prerequisite Physics: Conservation Laws", [
            "Elastic collisions conserve both momentum and energy.",
            "Momentum is the product of mass and velocity.",
            "Kinetic energy remains constant during these perfect bounces."
        ])
        
        # Define equations
        momentum_eq = MathTex("p = mv", color="#0000FF")
        energy_eq = MathTex("K = \\frac{1}{2}mv^2", color="#00FF00")
        
        # Define assets
        ball = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg", color=WHITE)
        pendulum = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pendulum.svg", color=WHITE)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#0000FF"))
        # Using columns 4-6 as per B004, vertical separation for B019
        self.place_at_grid(ball, 'B5', scale_factor=0.6)
        self.place_in_area(momentum_eq, 'B4', 'B4', scale_factor=1.0)
        momentum_eq.next_to(ball, RIGHT)
        self.play(FadeIn(ball), Write(momentum_eq))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        # Using columns 4-6, vertical separation
        self.place_at_grid(pendulum, 'E5', scale_factor=0.6)
        self.place_in_area(energy_eq, 'E4', 'E4', scale_factor=1.0)
        energy_eq.next_to(pendulum, RIGHT)
        self.play(FadeIn(pendulum), Write(energy_eq))
        self.wait(1)
