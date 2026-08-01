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
        title = "Scalars vs. Vectors: The Pirate's Map"
        lines = [
            "Scalars describe a quantity, like five miles.",
            "Vectors add direction, like five miles Northeast.",
            "We represent vectors as arrows with specific lengths."
        ]
        self.setup_layout(title, lines)

        # Assets
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pirate.svg]
        pirate = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pirate.svg")
        pirate.set_color(WHITE)
        pirate_label = Text("Captain Vector", font_size=18).next_to(pirate, DOWN, buff=0.1)
        pirate_group = VGroup(pirate, pirate_label)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/treasure.svg]
        chest = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/treasure.svg")
        chest.set_color("#FFD700") # Gold
        chest_label = Text("Treasure", font_size=18).next_to(chest, UP, buff=0.1)
        chest_group = VGroup(chest, chest_label)

        # Initial placement - Issue 23: Move pirate to D1. Issue 24: Move chest to B6.
        self.place_at_grid(pirate_group, "D1", scale_factor=0.6)
        self.place_at_grid(chest_group, "B6", scale_factor=0.6)

        self.add(pirate_group, chest_group)
        self.wait(1)

        # === Animation for Lecture Line 1 ===
        # Scalars describe a quantity, like five miles.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Circle representing 5 miles radius
        # Use a radius that fits visually within the grid area
        radius_circle = Circle(radius=1.5, color=YELLOW, stroke_width=2).move_to(pirate.get_center())
        radius_label = Text("5 miles", font_size=20, color=YELLOW).next_to(radius_circle, RIGHT, buff=0.1)
        
        self.play(Create(radius_circle), Write(radius_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Vectors add direction, like five miles Northeast.
        self.play(self.lecture[1].animate.set_color(BLUE))
        
        # Dashed line to treasure
        dashed_line = DashedLine(pirate.get_center(), chest.get_center(), color=BLUE_B)
        
        # Blue arrow
        vector_arrow = Arrow(
            start=pirate.get_center(), 
            end=chest.get_center(), 
            buff=0.3, 
            color="#0000FF", 
            stroke_width=6
        )
        
        self.play(
            FadeOut(radius_circle),
            FadeOut(radius_label),
            Create(dashed_line)
        )
        self.play(GrowArrow(vector_arrow))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We represent vectors as arrows with specific lengths.
        self.play(self.lecture[2].animate.set_color(GREEN))
        
        vector_label = Text("Vector: Magnitude & Direction", font_size=20, color=GREEN)
        # Issue 25: Place vector_label using place_in_area('A3', 'B5', scale_factor=0.6)
        self.place_in_area(vector_label, 'A3', 'B5', scale_factor=0.6)
        
        self.play(Write(vector_label))
        self.wait(2)
