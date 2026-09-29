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
            "Points, lines, and planes have classic dimensions.",
            "Some shapes defy these simple categories.",
            "Nature hides complexity in infinite detail."
        ]
        self.setup_layout("The Failure of Traditional Dimensions", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        line = Line(start=LEFT*1, end=RIGHT*1, color="#FFFFFF")
        # Ensure it is in column 4-6 per B002
        self.place_at_grid(line, 'C5', scale_factor=1.0)
        self.play(Create(line))
        self.lecture[0].set_color("#88AAFF")

        # === Animation for Lecture Line 2 ===
        square = Square(side_length=1, color="#FFFFFF")
        # Apply layout fixes from Critic (B002 compliant)
        self.place_in_area(square, 'B4', 'C6', scale_factor=1.0)
        self.play(Transform(line, square))
        self.lecture[1].set_color("#88AAFF")

        # === Animation for Lecture Line 3 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/fern.svg]
        # B018: Explicitly render and label
        fern = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fern.svg", color="#00FF00")
        fern_label = Text("Fern", font_size=16, color="#00FF00")
        
        # B002/B004/B003: peripheral quadrant
        self.place_at_grid(fern, 'E3', scale_factor=0.6)
        # B018: Label near object
        fern_label.next_to(fern, DOWN, buff=0.1)
        
        self.play(FadeIn(fern), Write(fern_label))
        
        # Fix from critic: poor grid utilization
        legend_label = Text("Fractal Growth", font_size=16, color="#FFFFFF")
        self.place_at_grid(legend_label, 'F2', scale_factor=0.5)
        
        self.play(FadeIn(legend_label))
        self.lecture[2].set_color("#88AAFF")
        self.wait(2)
