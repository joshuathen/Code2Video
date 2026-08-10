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
            "Heat equations model temperature flow over time.",
            "u_t = α ∇²u",
            "Change equals spatial curvature of temperature.",
            "Visualizing heat curvature.",
            "Heat naturally smooths out over time."
        ]
        self.setup_layout("The Heat Equation: A Core PDE", lecture_lines)
        
        # Assets
        thermometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/thermometer.svg")
        ice = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ice.svg")
        
        # === Animation for Lecture Line 1 ===
        # Heat equations model temperature flow over time.
        self.lecture[0].set_color(YELLOW)
        
        # === Animation for Lecture Line 2 ===
        # u_t = α ∇²u
        self.lecture[1].set_color("#00FFFF")
        eq = MathTex("u_t = \\\\alpha \\\\nabla^2 u", color="#00FFFF")
        self.place_in_area(eq, "A4", "B6", scale_factor=0.9)
        self.place_at_grid(thermometer, "A2", scale_factor=0.5)
        self.play(Write(eq), FadeIn(thermometer))

        # === Animation for Lecture Line 3 ===
        # Change equals spatial curvature of temperature.
        self.lecture[2].set_color(YELLOW)
        
        # === Animation for Lecture Line 4 ===
        # Visualizing heat curvature.
        self.lecture[3].set_color(GREEN)
        dot = Dot(color=GREEN)
        self.place_at_grid(dot, "E4", scale_factor=1.0)
        label = Text("Heat Flux", font_size=20).next_to(dot, RIGHT)
        self.play(FadeIn(dot), Write(label))

        # === Animation for Lecture Line 5 ===
        # Heat naturally smooths out over time.
        self.lecture[4].set_color(YELLOW)
        self.place_in_area(ice, "D1", "F3", scale_factor=0.8)
        self.play(dot.animate.set_color(BLUE), FadeIn(ice), run_time=2)
        self.wait(1)
