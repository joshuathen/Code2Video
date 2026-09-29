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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Real-World Applications", ["Refraction powers optical lenses.", "Lenses focus light precisely.", "This makes vision possible."])
        
        # === Animation for Lecture Line 1 ===
        # Refraction powers optical lenses.
        self.lecture[0].set_color("#FFD700")
        
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg", color="#87CEEB")
        self.place_at_grid(prism, "B6", scale_factor=0.6)
        
        light_ray = Line(RIGHT*2, RIGHT*4, color="#FFFFFF").shift(UP*1.0)
        bent_ray = Line(RIGHT*4, RIGHT*6, color="#FFFFFF").shift(UP*1.5)
        
        self.play(FadeIn(prism))
        self.play(Create(light_ray), Create(bent_ray))

        # === Animation for Lecture Line 2 ===
        # Lenses focus light precisely.
        self.lecture[1].set_color("#00FF00")
        
        lens = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lens.svg", color="#FF4500")
        self.place_at_grid(lens, "E5", scale_factor=0.5)
        
        ray1 = Line(RIGHT*2, RIGHT*6, color="#FF0000").shift(UP*0.3)
        ray2 = Line(RIGHT*2, RIGHT*6, color="#FF0000").shift(DOWN*0.3)
        
        self.play(FadeIn(lens), Create(ray1), Create(ray2))
        self.play(ray1.animate.shift(DOWN*0.2), ray2.animate.shift(UP*0.2))

        # === Animation for Lecture Line 3 ===
        # This makes vision possible.
        self.lecture[2].set_color("#1E90FF")
        
        eye = VGroup(
            Ellipse(width=1.5, height=1.0, color="#FFFFFF"),
            Circle(radius=0.3, color="#000000").set_fill("#000000", opacity=1)
        )
        self.place_at_grid(eye, "D4", scale_factor=0.4)
        self.play(FadeIn(eye))
