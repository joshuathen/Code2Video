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
        self.setup_layout("Boundary and Initial Conditions", [
            "PDEs need constraints to find unique solutions.",
            "Initial conditions define the state at time zero.",
            "Boundary conditions define behavior at the domain edges."
        ])
        
        # Elements
        # Using SVG asset
        pool_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pool.svg")
        domain_label = Text("PDE Solution Domain", font_size=18, color=WHITE)
        domain_group = VGroup(pool_icon, domain_label).arrange(DOWN, buff=0.1)
        
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 2, 1], axis_config={"include_numbers": False})
        interval_line = Line(axes.c2p(0, 0), axes.c2p(4, 0), color=WHITE)
        point_start = Dot(axes.c2p(0, 0), color="#FF00FF")
        point_end = Dot(axes.c2p(4, 0), color="#FF00FF")
        
        left_label = MathTex("u(0, t) = 0", font_size=24, color="#FF00FF")
        right_label = MathTex("u(L, t) = 0", font_size=24, color="#FF00FF")
        
        # Initial condition function
        initial_curve = axes.plot(lambda x: 1.5 * np.sin(np.pi * x / 4), color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.place_in_area(domain_group, "B2", "E5", scale_factor=0.6)
        self.play(FadeIn(domain_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        self.place_in_area(axes, "B3", "E6", scale_factor=0.5)
        self.place_in_area(initial_curve, "C2", "F5", scale_factor=0.6)
        self.play(FadeOut(domain_group), Create(axes), Create(interval_line), Create(point_start), Create(point_end), Create(initial_curve))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF00FF")
        self.place_at_grid(left_label, "D2", scale_factor=0.7)
        self.place_at_grid(right_label, "D5", scale_factor=0.7)
        self.play(Write(left_label), Write(right_label))
        self.wait(2)
