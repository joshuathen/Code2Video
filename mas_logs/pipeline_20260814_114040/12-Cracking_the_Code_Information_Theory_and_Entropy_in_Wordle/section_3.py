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
            "Maximize information gain to solve Wordle efficiently.",
            "Good words eliminate many incorrect candidates.",
            "Avoid words that rarely narrow the field.",
            "Compare average case against worst case scenarios.",
            "Information theory turns guessing into optimization."
        ]
        self.setup_layout("Strategy: Maximizing Information Gain", lecture_lines)
        
        # Assets
        dictionary = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dictionary.svg", color="#FFFFFF")
        pencil = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pencil.svg", color="#ADFF2F")
        
        # Elements
        target = dictionary
        self.place_at_grid(target, 'B5', scale_factor=0.6)
        
        node_1 = Dot(color="#FFFFFF")
        self.place_at_grid(node_1, 'D4', scale_factor=0.7)
        node_2 = Dot(color="#FFFFFF")
        self.place_at_grid(node_2, 'D6', scale_factor=0.7)
        node_3 = Dot(color="#FFFFFF")
        self.place_at_grid(node_3, 'E5', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#1E90FF")
        self.play(FadeIn(target), run_time=1.0)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#1E90FF")
        path1 = Line(node_1.get_center(), target.get_center(), color=WHITE)
        path2 = Line(node_2.get_center(), target.get_center(), color=WHITE)
        self.play(Create(path1), Create(path2), FadeIn(node_1, node_2), run_time=1.0)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#1E90FF")
        path3 = Line(node_3.get_center(), target.get_center(), color=GRAY)
        self.play(Create(path3), FadeIn(node_3), run_time=1.0)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#ADFF2F")
        self.play(pencil.animate.move_to(target.get_center()), run_time=1.0)
        self.play(path1.animate.set_color("#ADFF2F"), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#ADFF2F")
        self.play(Flash(target, color="#ADFF2F"), run_time=1.5)
        self.wait(2)
