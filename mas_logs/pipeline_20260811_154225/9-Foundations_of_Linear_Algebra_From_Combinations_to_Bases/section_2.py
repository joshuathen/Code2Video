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
        self.setup_layout("Redundancy: Linear Dependence", ["Redundant vectors add no new reach.", "They lie on the same line.", "This redundancy is linear dependence."])
        
        # Vector icons asset
        vectors_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vectors.svg")
        
        # Vectors setup
        v1 = Vector([0, 1.5, 0], color=BLUE)
        v2 = Vector([1.5, 0, 0], color=GREEN)
        v3 = Vector([1.5, 1.5, 0], color=YELLOW)
        
        vector_group = VGroup(v1, v2, v3, vectors_icon)
        
        # B004 & B021: Place vector_group in C3-E6 (right side of center)
        self.place_in_area(vector_group, 'C3', 'E6', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE), Create(v1), Create(v2), Create(v3), FadeIn(vectors_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        v3_dashed = DashedLine(start=ORIGIN, end=v3.get_end(), color=YELLOW)
        
        self.play(self.lecture[1].animate.set_color(GREEN), Transform(v3, v3_dashed))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        label = Text("Redundant", color="#FF33A8")
        label.scale(0.7) # B020
        # B011: Tethering
        self.add(label)
        label.next_to(v3, UP)
        
        self.play(self.lecture[2].animate.set_color("#FF33A8"), Write(label))
        self.wait(2)
