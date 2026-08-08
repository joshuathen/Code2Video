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
            "ODEs track change in one variable.",
            "PDEs model change in many variables.",
            "Think of space and time together.",
            "Bouncing ball follows an ODE.",
            "Vibrating drums follow a PDE."
        ]
        self.setup_layout("From ODEs to PDEs: A Visual Shift", lecture_lines)
        
        # Elements
        ball = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg")
        drum = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drum.svg")
        wave = FunctionGraph(lambda x: 0.5 * np.sin(3 * x), color=GREEN)
        surface = Surface(
            lambda u, v: np.array([u, v, 0.2 * np.sin(3 * u) * np.cos(3 * v)]),
            u_range=[-1.5, 1.5], v_range=[-1.5, 1.5],
            fill_opacity=0.6, color=YELLOW
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_at_grid(ball, 'B5', scale_factor=0.5)
        self.play(Create(ball))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        self.place_at_grid(wave, 'C5', scale_factor=0.6)
        self.play(FadeIn(wave))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.place_in_area(surface, 'B4', 'E6', scale_factor=0.6)
        self.play(FadeIn(surface))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(BLUE)
        self.play(Indicate(ball)) 

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(YELLOW)
        self.place_at_grid(drum, 'F3', scale_factor=0.5)
        self.play(FadeIn(drum), Indicate(surface)) 
        self.wait(2)
