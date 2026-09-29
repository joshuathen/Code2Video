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
            "The heat equation models thermal diffusion.",
            "Temperature distributes along the metal rod.",
            "Natural modes decay over time."
        ]
        self.setup_layout("Bridging to the Heat Equation", lecture_lines)
        
        # Define objects
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg]
        rod = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg")
        
        # Animation 1
        func_u = FunctionGraph(lambda x: 0.5 * np.sin(np.pi * x) + 0.2 * np.sin(3 * np.pi * x), x_range=[-1, 1], color="#FFFFFF")
        
        # Animation 2
        heat_label = Text("Heat Flow", font_size=20, color="#E67E22")
        equation = MathTex(r"\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}", color="#E67E22")
        
        # Animation 3
        diffusion_label = Text("Diffusion", font_size=20, color="#3498DB")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_in_area(rod, "A2", "C4", scale_factor=0.6)
        self.place_in_area(func_u, "A2", "C4", scale_factor=0.6)
        self.play(FadeIn(rod), Create(func_u))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#E67E22"))
        self.place_at_grid(equation, "D3", scale_factor=0.9)
        self.place_at_grid(heat_label, "E3", scale_factor=0.7)
        self.play(Write(equation), Write(heat_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#3498DB"))
        self.place_at_grid(diffusion_label, "F3", scale_factor=0.7)
        self.play(Write(diffusion_label))
        
        # Animate function flattening
        target_func = FunctionGraph(lambda x: 0.2 * np.sin(np.pi * x), x_range=[-1, 1], color="#3498DB")
        self.play(Transform(func_u, target_func))
        self.wait(2)
