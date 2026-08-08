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
        self.setup_layout("Linear Dependence vs. Independence", [
            "Redundant vectors are linearly dependent.",
            "Independent vectors add new dimensions.",
            "Redundancy collapses dimensions in space."
        ])
        
        # Vectors and Assets
        v1 = Vector([1, 1], color=BLUE)
        v2 = Vector([0.5, 0.5], color=BLUE)
        v3 = Vector([-0.5, 0.5], color=YELLOW)
        
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        
        vector_group = VGroup(v1, v2, ruler)
        vector_animation = self.place_in_area(vector_group, 'B3', 'E5', scale_factor=0.6)
        
        grid_title = Text("Vector Space Analysis", font_size=20)
        self.place_at_grid(grid_title, 'A3', scale_factor=0.9)
        
        svm_text = Text("Redundancy = 0D collapse", font_size=18)
        self.place_in_area(svm_text, 'C1', 'D3', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(vector_animation))
        self.lecture[0].set_color("#FF0000")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(v3), FadeIn(protractor))
        self.lecture[1].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        self.play(FadeIn(svm_text))
        self.wait(2)
