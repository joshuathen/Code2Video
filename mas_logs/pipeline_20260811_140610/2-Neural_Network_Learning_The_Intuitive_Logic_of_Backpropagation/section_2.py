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
        self.setup_layout("Prerequisite: The Loss Surface", [
            "Errors create a hilly landscape called loss.",
            "We seek the lowest valley for minimal error.",
            "Gradient descent helps us navigate toward it."
        ])
        
        # === Animation for Lecture Line 1 ===
        axes = ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2], z_range=[0, 3], axis_config={"include_tip": False})
        surface = Surface(
            lambda u, v: np.array([u, v, 0.5 * (np.sin(2 * u) + np.cos(2 * v)) + 1]),
            u_range=[-2, 2], v_range=[-2, 2],
            resolution=(20, 20),
            fill_opacity=0.7,
            stroke_width=0.5,
            stroke_color=BLUE
        ).set_color("#00FFFF")
        
        mountain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mountain.svg").set_color(WHITE)
        valley = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/valley.svg").set_color(WHITE)
        
        landscape = VGroup(axes, surface, mountain, valley)
        # Applying fix for Issue 23
        self.place_in_area(landscape, 'A2', 'D4', scale_factor=0.5)
        
        self.play(Create(landscape))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        dot = Dot(color=YELLOW)
        label = Text("Current Loss", font_size=16, color=YELLOW)
        dot_group = VGroup(dot, label).arrange(UP, buff=0.1)
        # Applying fix for Issue 24
        self.place_at_grid(dot_group, 'E3', scale_factor=0.7)
        
        self.play(FadeIn(dot_group))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        arrow = Arrow(start=UP*0.5, end=ORIGIN, color=RED, buff=0)
        # Applying fix for Issue 25
        self.place_at_grid(arrow, 'E4', scale_factor=0.8)
        
        self.play(GrowArrow(arrow))
        self.lecture[2].set_color(RED)
        
        self.wait(2)
