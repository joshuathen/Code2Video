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
        self.setup_layout("Real-World Application: The Archer Fish", [
            "The Archer Fish hunts above water.",
            "Refraction shifts the appearance of prey.",
            "The fish aims below the visible target."
        ])
        
        # Setup static elements
        water = Rectangle(width=6, height=1, color="#1E90FF").set_fill("#1E90FF", opacity=0.5)
        self.place_in_area(water, "E1", "F6", scale_factor=0.9)
        self.add(water)

        fish = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fish.svg")
        fish_label = Text("Archer Fish", font_size=16, color="#FFA500")
        fish_group = VGroup(fish, fish_label).arrange(DOWN)
        self.place_at_grid(fish_group, "D2", scale_factor=0.3)
        self.add(fish_group)

        target = Circle(radius=0.2, color="#00FF00").set_fill("#00FF00", opacity=1)
        target_label = Text("Prey", font_size=16, color="#00FF00")
        target_group = VGroup(target, target_label).arrange(UP)
        self.place_at_grid(target_group, "A4", scale_factor=0.7)
        self.add(target_group)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFA500")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        # Visualizing refraction
        shifted_target = Circle(radius=0.2, color="#FF0000").set_fill("#FF0000", opacity=1)
        shifted_label = Text("Shifted Target", font_size=16, color="#FF0000")
        shifted_group = VGroup(shifted_target, shifted_label).arrange(UP)
        self.place_at_grid(shifted_group, "B4", scale_factor=0.7)
        self.play(FadeIn(shifted_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#0000FF")
        jet = Line(start=self.grid["D2"], end=self.grid["A4"], color="#0000FF", stroke_width=4)
        jet_label = Text("Water Jet", font_size=16, color="#0000FF")
        self.place_at_grid(jet_label, "D3", scale_factor=0.8)
        self.play(Create(jet), Write(jet_label))
        self.wait(2)
