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
        lecture_lines = [
            "The fox scenario illustrates Bayesian updating.",
            "Start with a prior belief about location.",
            "Incorporate rustling sounds as new evidence.",
            "Calculate the updated probability of the zone.",
            "Bayes' helps us reason with uncertain data."
        ]
        self.setup_layout("Practical Application: The 'Smart Fox' Scenario", lecture_lines)
        
        # Load assets
        fox = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fox.svg")
        fox.set_color("#FF8C00")
        observation = Circle(color="#00FF00", radius=0.3)
        
        # Initialize line with an updater
        line = Line(start=ORIGIN, end=ORIGIN, color=WHITE)
        line.add_updater(lambda m: m.put_start_and_end_on(fox.get_center(), observation.get_center()))
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF8C00"))
        self.place_at_grid(fox, 'B4', scale_factor=0.5)
        self.play(FadeIn(fox))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF8C00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.place_at_grid(observation, 'E3', scale_factor=0.9)
        self.play(FadeIn(observation), Create(line))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF8C00"))
        self.play(Indicate(fox))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        self.wait(2)
