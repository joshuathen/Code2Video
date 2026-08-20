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
            "Think of the sphere as thin horizontal rings.",
            "Project these rings onto the circumscribing cylinder.",
            "The cylinder's area equals the sphere's area.",
            "Everything sums up to four shadow areas.",
            "Calculus confirms the geometric proof."
        ]
        self.setup_layout("The Calculus Logic", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Display a smooth curve with #FFFF00.
        curve = FunctionGraph(lambda x: 0.5 * np.sin(x*2) + 1, x_range=[-1.5, 1.5], color="#FFFF00")
        self.place_in_area(curve, 'B4', 'E6', scale_factor=0.6) # Per issue 23/31
        self.play(Create(curve))
        self.lecture[0].set_color("#FFFF00")

        # === Animation for Lecture Line 2 ===
        # Draw tiny vertical rectangles using cylinder asset
        cylinder_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cylinder.svg")
        
        rects = VGroup()
        for x in np.linspace(-1.0, 1.0, 5):
            h = 0.5 * np.sin(x*2) + 1
            # Using the cylinder asset as a placeholder/indicator
            rect = cylinder_asset.copy()
            rect.set_height(h * 0.8)
            rect.set_color("#00FFFF")
            rect.move_to(curve.point_from_proportion((x+1.5)/3.0) - np.array([0, h/2, 0]))
            rects.add(rect)
        self.play(LaggedStart(*[FadeIn(r) for r in rects], lag_ratio=0.1))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF8800")
        self.wait(2)
