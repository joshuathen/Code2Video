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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Complex waves are like blended fruit smoothies.",
            "We can extract pure sine wave ingredients.",
            "This decomposition is a Fourier Series."
        ]
        self.setup_layout("The Intuitive Hook: The 'Smoothie' Analogy", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        blender = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blender.svg", color=WHITE)
        smoothie = FunctionGraph(lambda x: 0.5 * np.sin(x) + 0.3 * np.sin(2*x) + 0.2 * np.sin(3*x), x_range=[-PI, PI], color="#008B8B")
        
        group1 = VGroup(blender, smoothie).arrange(DOWN)
        self.place_in_area(group1, 'A4', 'B6', scale_factor=0.7)
        self.play(FadeIn(blender), Create(smoothie))
        self.lecture[0].set_color("#008B8B")

        # === Animation for Lecture Line 2 ===
        fruit = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fruit.svg", color=WHITE)
        sine1 = FunctionGraph(lambda x: 0.5 * np.sin(x), x_range=[-PI, PI], color="#FF8C00")
        sine2 = FunctionGraph(lambda x: 0.3 * np.sin(2*x), x_range=[-PI, PI], color="#EE82EE")
        sine3 = FunctionGraph(lambda x: 0.2 * np.sin(3*x), x_range=[-PI, PI], color="#32CD32")
        
        # Applying requested grid utilization
        self.place_in_area(sine1, 'D1', 'F2', scale_factor=0.6)
        self.place_in_area(sine2, 'D3', 'F4', scale_factor=0.6)
        self.place_in_area(sine3, 'D5', 'F6', scale_factor=0.6)
        
        self.play(
            FadeOut(smoothie), FadeOut(blender),
            Create(sine1), Create(sine2), Create(sine3),
            FadeIn(fruit)
        )
        self.lecture[1].set_color("#FF8C00")

        # === Animation for Lecture Line 3 ===
        # Using placeholder for now as requested string object logic
        string = Line(start=np.array([-1, 0, 0]), end=np.array([1, 0, 0]), color=WHITE)
        self.place_at_grid(string, 'C3', scale_factor=1.0)
        
        self.play(
            FadeOut(sine1), FadeOut(sine2), FadeOut(sine3), FadeOut(fruit),
            Create(string)
        )
        self.lecture[2].set_color(WHITE)
