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
            "Visualize the Cost Function as a landscape.",
            "Low points represent minimal error predictions.",
            "The algorithm acts like a hiker.",
            "It descends towards the lowest valley.",
            "[Asset: error_surface] displays the error landscape."
        ]
        
        self.setup_layout("The Error Surface Visualization", lecture_lines)
        
        # Elements
        surface = Surface(
            lambda u, v: np.array([u, v, 0.5 * (u**2 + v**2)]),
            u_range=[-1.5, 1.5], v_range=[-1.5, 1.5],
            resolution=(15, 15)
        ).set_fill(opacity=0.5).set_stroke(color=WHITE, width=0.5)
        
        hiker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hiker.svg", color="#00FF00")
        min_point = Dot(color="#FFFF00")
        gradient_vector = Arrow(start=ORIGIN, end=RIGHT*0.5, color="#FF4500")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.place_in_area(surface, 'B3', 'E6', scale_factor=0.55)
        self.play(Create(surface))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        self.place_at_grid(min_point, 'E4')
        self.play(FadeIn(min_point))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        self.place_at_grid(hiker, 'A2', scale_factor=0.8)
        self.play(FadeIn(hiker))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF4500")
        self.place_at_grid(gradient_vector, 'C5', scale_factor=0.8)
        self.play(GrowArrow(gradient_vector))
        self.play(hiker.animate.move_to(self.grid['E4']))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#808080")
        self.play(FadeOut(gradient_vector), surface.animate.set_stroke(color="#808080", opacity=0.3))
