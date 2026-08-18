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
        self.setup_layout("The Hook: How many slices can you get?", [
            "Place dots on a circle's edge.",
            "Connect every pair with a chord.",
            "Count the regions created inside."
        ])
        
        # Assets
        pizza = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pizza.svg", color="#FFD700")
        knife = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/knife.svg", color="#FFFFFF")
        
        # Geometric Construction Group
        construction = VGroup()
        self.place_in_area(pizza, 'B2', 'D4', scale_factor=1.2)
        construction.add(pizza)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.play(FadeIn(pizza))
        
        dot = Dot(color=WHITE)
        self.place_at_grid(dot, 'C3', scale_factor=1.5)
        construction.add(dot)
        self.play(FadeIn(dot))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        # Creating a single vertical cut
        cut1 = knife.copy().rotate(PI/2)
        self.place_at_grid(cut1, 'C3', scale_factor=0.5)
        self.play(FadeIn(cut1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFFFF")
        # Creating a second horizontal cut
        cut2 = knife.copy()
        self.place_at_grid(cut2, 'C3', scale_factor=0.5)
        self.play(FadeIn(cut2))
        self.wait(1)
