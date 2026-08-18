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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Synthesis", [
            "Euler's characteristic is an invariant property.",
            "Dual graphs provide deep reciprocal geometric insight.",
            "Geometric space limits network connectivity possibilities."
        ])
        
        # Colors
        c1, c2, c3 = "#FFD700", "#00BFFF", "#FF4500"

        # === Animation for Lecture Line 1 ===
        # Euler's characteristic: V - E + F = 2
        formula = MathTex(r"V - E + F = 2", font_size=48, color=c1)
        self.place_at_grid(formula, 'B2')
        self.play(Write(formula))
        self.lecture[0].set_color(c1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Dual graph construction
        # Simple representation: A square with a diagonal and its dual vertex in each face
        graph = VGroup(
            Line(self.grid['C4'], self.grid['C6']),
            Line(self.grid['C6'], self.grid['E6']),
            Line(self.grid['E6'], self.grid['E4']),
            Line(self.grid['E4'], self.grid['C4']),
            Line(self.grid['C4'], self.grid['E6'])
        ).set_stroke(c2, 4)
        
        dual_nodes = VGroup(
            Dot(self.grid['D5'], color=c2),
            Dot(self.grid['D6'], color=c2), # placeholder for faces
            Dot(self.grid['D4'], color=c2)
        )
        # Visualizing a simplified dual connection
        self.play(Create(graph), FadeIn(dual_nodes))
        self.lecture[1].set_color(c2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Geometric space limit
        circle = Circle(radius=1.2, color=c3).set_stroke(width=3)
        self.place_at_grid(circle, 'E3')
        label = Text("Geometric Constraints", font_size=20, color=c3)
        self.place_at_grid(label, 'F3')
        
        self.play(Create(circle), Write(label))
        self.lecture[2].set_color(c3)
        self.wait(2)
