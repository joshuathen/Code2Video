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
        self.setup_layout("Prerequisite: The Concept of Distributions", [
            "Data shapes aren't always bell-shaped.",
            "Some distributions are uniform or skewed.",
            "Most real-world data is messy."
        ])
        
        # Assets
        container_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/container.svg")
        mountain_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mountain.svg")
        
        # Colors
        COLOR_U = "#FF5733"
        COLOR_N = "#33FF57"
        COLOR_S = "#FF33FF"

        # === Animation for Lecture Line 1 ===
        # Data shapes aren't always bell-shaped.
        uniform_box = container_icon.copy().set_color(COLOR_U)
        normal_curve = mountain_icon.copy().set_color(COLOR_N)
        
        group_dist = VGroup(uniform_box, normal_curve).arrange(RIGHT, buff=1)
        self.place_in_area(group_dist, 'A2', 'C5', scale_factor=0.6)
        
        label_u = Text("Uniform", color=COLOR_U, font_size=20).next_to(uniform_box, UP)
        label_n = Text("Normal", color=COLOR_N, font_size=20).next_to(normal_curve, UP)
        
        self.play(FadeIn(uniform_box), FadeIn(normal_curve), Write(label_u), Write(label_n))
        self.lecture[0].set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Some distributions are uniform or skewed.
        skewed_curve = FunctionGraph(lambda x: (x+2)**2 * np.exp(-x-2), x_range=[-2, 3], color=COLOR_S)
        self.place_at_grid(skewed_curve, 'D3', scale_factor=0.5)
        label_s = Text("Skewed", color=COLOR_S, font_size=20).next_to(skewed_curve, UP)
        
        self.play(Create(skewed_curve), Write(label_s))
        self.lecture[1].set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Most real-world data is messy.
        dots = VGroup(*[Dot(color=WHITE, radius=0.03) for _ in range(50)])
        for dot in dots:
            dot.move_to(self.grid['E5'] + np.random.uniform(-0.5, 0.5, 3))
        
        real_world_text = Text("Real-world data", color=WHITE, font_size=24)
        self.place_at_grid(real_world_text, 'E5', scale_factor=0.7)
        
        self.play(FadeIn(dots), Write(real_world_text))
        self.lecture[2].set_color("#FFD700")
        self.wait(2)
