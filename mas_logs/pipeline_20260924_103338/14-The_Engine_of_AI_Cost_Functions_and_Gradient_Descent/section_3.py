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
            "Gradient Descent helps us find the valley floor.",
            "The gradient points toward the steepest ascent.",
            "We move in the opposite, downhill direction."
        ]
        self.setup_layout("The Mechanism: Gradient Descent", lecture_lines)
        
        # Load Assets
        valley_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/valley.svg")
        self.place_at_grid(valley_icon, "E5", scale_factor=0.3)
        
        # Parabola (Cost function valley)
        valley_graph = FunctionGraph(lambda x: 0.2 * x**2 - 1.5, x_range=[-2, 2], color=BLUE)
        self.place_in_area(valley_graph, "C2", "E6", scale_factor=0.9)
        
        # Hiker/Parameters dot
        hiker = Dot(color=WHITE)
        hiker.move_to(self.grid["B5"])
        
        # Target (Valley bottom)
        target = Dot(color=GREEN, radius=0.15)
        self.place_at_grid(target, "C4", scale_factor=0.7)
        
        self.add(valley_graph, valley_icon, hiker, target)

        # Labels
        hiker_label = Text("Parameters", font_size=18, color=WHITE).next_to(hiker, UP, buff=0.1)
        valley_label = Text("Valley", font_size=18, color=GREEN).next_to(valley_icon, DOWN, buff=0.1)
        self.add(hiker_label, valley_label)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#88CCFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF5555"))
        
        # Gradient vector pointing uphill
        grad_vec = Arrow(start=hiker.get_center(), end=self.grid["B6"], color="#FF5555", buff=0)
        self.play(Create(grad_vec))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # Move in opposite direction
        step = Arrow(start=hiker.get_center(), end=target.get_center(), color="#FFFF00", buff=0)
        self.play(
            hiker.animate.move_to(target.get_center()),
            Create(step),
            run_time=1.5
        )
        
        self.wait(2)
