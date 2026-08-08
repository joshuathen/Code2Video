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
        self.setup_layout("Prerequisite: The Cost Function (The 'Error Map')", [
            "Visualize error as a mountain landscape.",
            "The valley floor is minimum error.",
            "Our goal is to reach the valley.",
            "Elevation represents the total prediction error.",
            "Weights act as our location coordinates."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Visualize error as a mountain landscape.
        self.lecture[0].set_color("#00FF00")
        mountain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mountain.svg")
        mountain.set_color("#00FF00")
        self.place_at_grid(mountain, 'C4', scale_factor=0.7)
        self.play(FadeIn(mountain))

        # === Animation for Lecture Line 2 ===
        # The valley floor is minimum error.
        self.lecture[1].set_color("#00FF00")
        valley = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/valley.svg")
        valley.set_color("#0000FF")
        self.place_at_grid(valley, 'E4', scale_factor=0.5)
        self.play(FadeIn(valley))

        # === Animation for Lecture Line 3 ===
        # Our goal is to reach the valley.
        self.lecture[2].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Elevation represents the total prediction error.
        self.lecture[3].set_color("#FFFF00")
        height_bar = Line(start=ORIGIN, end=UP*1, color="#FFFF00")
        self.place_at_grid(height_bar, 'D5', scale_factor=1.0)
        self.play(Create(height_bar))

        # === Animation for Lecture Line 5 ===
        # Weights act as our location coordinates.
        self.lecture[4].set_color("#FFFFFF")
        weight_label = Text("Weights", font_size=18, color="#FFFFFF")
        self.place_at_grid(weight_label, 'F5', scale_factor=1.0)
        self.play(Write(weight_label))
        self.wait(2)
