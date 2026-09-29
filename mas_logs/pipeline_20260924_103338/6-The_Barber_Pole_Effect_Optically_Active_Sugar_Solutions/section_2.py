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
        lecture_lines = ["Chiral molecules act on polarized light.", "Sugar solutions physically twist the polarization plane.", "This rotation is called optical activity."]
        self.setup_layout("The Phenomenon of Optical Rotation", lecture_lines)
        
        # Define mobjects
        solution = Rectangle(width=2, height=4, color="#8BC34A", fill_opacity=0.3)
        self.place_in_area(solution, 'B3', 'D4', scale_factor=0.6)
        
        sol_label = Text("Solution", font_size=24, color="#8BC34A")
        self.place_at_grid(sol_label, 'B3', scale_factor=0.7)

        light_wave = VGroup(*[Line(UP*0.5, DOWN*0.5, color="#FF9800") for _ in range(10)]).arrange(RIGHT, buff=0.2)
        self.place_in_area(light_wave, 'B3', 'D4', scale_factor=0.6)

        theta = MathTex(r"\\theta", color="#E91E63")
        self.place_at_grid(theta, 'C5', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(solution), Write(sol_label))
        self.lecture[0].set_color("#8BC34A")
        self.play(Create(light_wave))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF9800")
        self.play(Rotate(light_wave, angle=PI/4, about_point=self.grid["C3"]))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#E91E63")
        self.play(Write(theta))
        self.wait(1)
