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
        lecture_lines = [
            "Parallel rays create a cylindrical shadow.",
            "The sphere's shadow is a circle of radius r.",
            "The shadow's area is exactly πr²."
        ]
        self.setup_layout("The Geometry of the Projected Shadow", lecture_lines)
        
        # Elements
        sphere_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color=GOLD).set_opacity(0.8)
        shadow = Circle(radius=1, color=GRAY, fill_opacity=0.5, stroke_width=2)
        label_r = MathTex("r", color=WHITE)

        # Setup
        scene_group = VGroup(sphere_icon, shadow)
        self.place_at_grid(scene_group, 'C5', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.play(Create(sphere_icon), run_time=1.5)
        self.lecture[0].set_color("#FFFFE0")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.play(Create(shadow), run_time=1.5)
        self.place_at_grid(label_r, 'D6', scale_factor=0.6)
        self.play(FadeIn(label_r))
        self.lecture[1].set_color("#808080")

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        area_formula = MathTex(r"A = \pi r^2", color=WHITE)
        self.place_at_grid(area_formula, 'E5', scale_factor=0.8)
        self.play(Write(area_formula))
        self.lecture[2].set_color("#FF4500")
        
        self.wait(2)
