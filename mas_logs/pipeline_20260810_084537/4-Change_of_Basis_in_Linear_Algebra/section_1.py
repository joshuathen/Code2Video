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
        self.setup_layout("Prerequisite & Intuition: The 'Language' of Coordinates", [
            "A basis acts as your coordinate system's grid. [Asset: grid_01]",
            "Standard vectors i and j define your directions. [Asset: vectors_01]",
            "Point (3, 2) is a simple set of instructions. [Asset: point_01]"
        ])
        
        # Create elements
        grid_01 = NumberPlane(x_range=[-1, 4, 1], y_range=[-1, 4, 1])
        
        e1 = Arrow(start=ORIGIN, end=RIGHT, color="#FF0000")
        e2 = Arrow(start=ORIGIN, end=UP, color="#FF0000")
        vectors_01 = VGroup(e1, e2)
        
        point_01 = Dot(color="#00FF00")
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(grid_01, 'B2', 'E5', scale_factor=0.6)
        self.play(FadeIn(grid_01))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Anchor basis to grid center
        vectors_01.move_to(grid_01.get_center())
        self.place_at_grid(vectors_01, 'C3', scale_factor=0.4)
        
        self.play(Create(vectors_01))
        self.lecture[1].set_color("#FF0000")

        # === Animation for Lecture Line 3 ===
        dashed_x = DashedLine(grid_01.c2p(3, 0), grid_01.c2p(3, 2), color="#FFFF00")
        dashed_y = DashedLine(grid_01.c2p(0, 2), grid_01.c2p(3, 2), color="#FFFF00")
        
        point_01.move_to(grid_01.c2p(3, 2))
        self.place_at_grid(point_01, 'D4', scale_factor=0.5)
        
        self.play(FadeIn(point_01))
        self.play(Create(dashed_x), Create(dashed_y))
        self.lecture[2].set_color("#00FF00")
        
        self.wait(1)
        self.play(FadeOut(dashed_x), FadeOut(dashed_y), FadeOut(grid_01), FadeOut(vectors_01), FadeOut(point_01))
