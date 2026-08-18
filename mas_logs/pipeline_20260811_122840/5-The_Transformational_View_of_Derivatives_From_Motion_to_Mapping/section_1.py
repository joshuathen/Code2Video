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
        self.setup_layout("Prerequisite: The Concept of a 'Function as a Map'", [
            "Functions transform inputs into outputs.",
            "Imagine a rubber band stretching.",
            "Points map to new positions.",
            "The derivative measures local density.",
            "Transformations change spatial relationships."
        ])
        
        # Mobjects
        set_A = Circle(radius=0.8, color=WHITE).set_fill(opacity=0.1)
        set_B = Circle(radius=0.8, color=WHITE).set_fill(opacity=0.1)
        label_A = Text("Set A", font_size=24, color=WHITE).next_to(set_A, UP)
        label_B = Text("Set B", font_size=24, color=WHITE).next_to(set_B, UP)
        
        mapping_group = VGroup(set_A, set_B, label_A, label_B)
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(mapping_group, "A4", "C6", scale_factor=0.6)
        self.play(FadeIn(mapping_group))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#3498DB")
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#3498DB")
        x_pt = Dot(set_A.get_center(), color="#F1C40F")
        fx_pt = Dot(set_B.get_center(), color="#F1C40F")
        arrow = Arrow(x_pt.get_center(), fx_pt.get_center(), buff=0.1, color="#3498DB")
        
        self.play(Create(x_pt), Create(fx_pt), GrowArrow(arrow))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#E74C3C")
        label_deriv = Text("Derivative = Density", font_size=20, color="#E74C3C")
        self.place_at_grid(label_deriv, "E5", scale_factor=0.7)
        self.play(Write(label_deriv))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#2ECC71")
        self.play(mapping_group.animate.set_color("#2ECC71"), run_time=1.5)
