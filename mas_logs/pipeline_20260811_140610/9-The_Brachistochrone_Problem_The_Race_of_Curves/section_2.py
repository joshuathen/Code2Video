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
            "Gravity converts potential energy into kinetic energy.",
            "Steeper drops build speed much faster.",
            "Balancing distance and speed is key."
        ]
        self.setup_layout("The Brachistochrone Problem: The Race of Curves", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Gravity converts potential energy into kinetic energy.
        self.lecture[0].set_color("#88CCEE")
        
        axes = Axes(x_range=[0, 3, 1], y_range=[0, 3, 1], axis_config={"include_numbers": False})
        self.place_in_area(axes, "B3", "E5", scale_factor=0.5)
        self.play(Create(axes))

        # === Animation for Lecture Line 2 ===
        # Steeper drops build speed much faster.
        self.lecture[1].set_color("#FFCC66")
        
        point_A = Dot(color="#FFFF33").move_to(axes.c2p(0, 2))
        point_B = Dot(color="#FFFF33").move_to(axes.c2p(2, 0))
        
        # Assets
        marble = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/marble.svg")
        self.place_at_grid(marble, "B4", scale_factor=0.3)
        marble.move_to(point_A.get_center())

        label_A = Text("A", font_size=20).scale(0.7)
        label_A.next_to(point_A, UP)
        label_B = Text("B", font_size=20).scale(0.7)
        label_B.next_to(point_B, DOWN)
        
        self.play(FadeIn(point_A, label_A, marble), FadeIn(point_B, label_B))

        # === Animation for Lecture Line 3 ===
        # Balancing distance and speed is key.
        self.lecture[2].set_color("#FF99CC")
        
        time_text = MathTex("T(path) = \\int_{A}^{B} \\frac{ds}{\\sqrt{2gy}}", color="#FFFFFF")
        self.place_at_grid(time_text, "B6", scale_factor=0.7)
        
        rollercoaster = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rollercoaster.svg")
        self.place_at_grid(rollercoaster, "D4", scale_factor=0.5)
        
        self.play(Write(time_text), FadeIn(rollercoaster))
        self.wait(2)
