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
        lecture_lines = [
            "The Hilbert curve improves on Peano.",
            "It optimizes spatial locality effectively.",
            "Nearby 2D points stay close in 1D."
        ]
        self.setup_layout("The Hilbert Curve and Locality", lecture_lines)
        
        # Grid from asset
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        
        # 1. Draw a 2D plane with an 8x8 grid (Color #FFFFFF)
        # Using Critic advice for positioning
        self.place_at_grid(grid_asset.copy(), 'D3', scale_factor=0.35)
        grid_asset.set_color("#FFFFFF")

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(grid_asset))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Trace the Hilbert path (Color #00FF00)
        # Using Critic advice for positioning
        path = VMobject()
        path.set_points_as_corners([
            np.array([0.5, 0.5, 0]), np.array([0.5, 7.5, 0]), np.array([1.5, 7.5, 0]), np.array([1.5, 0.5, 0]),
            np.array([2.5, 0.5, 0]), np.array([2.5, 7.5, 0]), np.array([3.5, 7.5, 0]), np.array([3.5, 0.5, 0]),
            np.array([4.5, 0.5, 0]), np.array([4.5, 7.5, 0]), np.array([5.5, 7.5, 0]), np.array([5.5, 0.5, 0]),
            np.array([6.5, 0.5, 0]), np.array([6.5, 7.5, 0]), np.array([7.5, 7.5, 0]), np.array([7.5, 0.5, 0])
        ])
        path.set_stroke(color="#00FF00", width=4)
        self.place_in_area(path, 'B4', 'F6', scale_factor=0.5)
        
        self.play(Create(path), run_time=2)
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        # Highlight spatial proximity (Color #FF00FF)
        highlight = Circle(radius=0.3, color="#FF00FF").shift(self.grid['B3'])
        highlight2 = Circle(radius=0.3, color="#FF00FF").shift(self.grid['C3'])
        
        self.play(FadeIn(highlight), FadeIn(highlight2))
        self.lecture[2].set_color("#FF00FF")
        self.wait(2)
