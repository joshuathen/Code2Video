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
        self.setup_layout("Mathematical Formulation of Time", [
            "Total time depends on distance and speed.",
            "Speed relates to the index of refraction.",
            "Express total time using boundary distance x."
        ])

        # Assets
        stopwatch = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/stopwatch.svg")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")

        # === Animation for Lecture Line 1 ===
        # Total time depends on distance and speed.
        time_eq = MathTex("T", "=", "\\frac{d_1}{v_1}", "+", "\\frac{d_2}{v_2}")
        self.place_in_area(time_eq, 'B3', 'C5', scale_factor=0.85)
        self.place_at_grid(stopwatch, 'B2', scale_factor=0.4)
        
        self.play(Write(time_eq), FadeIn(stopwatch))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Speed relates to the index of refraction.
        # Color T #FFFFFF, distance #FF4500, v #00FF00.
        time_eq.set_color_by_tex("T", WHITE)
        time_eq.set_color_by_tex("d", "#FF4500")
        time_eq.set_color_by_tex("v", "#00FF00")
        
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FF00")
        self.play(FadeIn(self.lecture[1]))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Express total time using boundary distance x.
        self.lecture[2].set_color("#FF4500")
        x_label = MathTex("x", "=", "\\text{distance along boundary}")
        self.place_at_grid(x_label, 'D3', scale_factor=0.85)
        self.place_at_grid(ruler, 'D5', scale_factor=0.4)
        
        self.play(Write(x_label), FadeIn(self.lecture[2]), FadeIn(ruler))
        self.wait(2)
