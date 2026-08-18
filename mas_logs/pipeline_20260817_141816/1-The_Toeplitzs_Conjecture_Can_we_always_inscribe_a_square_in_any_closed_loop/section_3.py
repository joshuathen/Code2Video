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
            "Borsuk-Ulam theorem is our logical bridge.",
            "It links sphere points to plane values.",
            "It proves existence by mapping to zero.",
            "Symmetry helps us identify the square's corners.",
            "This confirms a square must always exist."
        ]
        self.setup_layout("Mathematical Core: Symmetry and the Borsuk-Ulam Theorem", lecture_lines)

        # Assets
        # Using SVG directly as requested in storyboard
        # Since SVGMobject is "expensive" but necessary for the asset requirement, we load once
        # Using a simple circle as a fallback placeholder if asset loading fails in test
        try:
            torus_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/torus.svg")
        except:
            torus_asset = Circle(radius=1.0, color=WHITE)
        
        torus_label = Text("Torus Surface", font_size=20, color=WHITE)
        
        # Grid positioning based on critics
        self.place_in_area(torus_asset, 'B4', 'E6', scale_factor=1.2)
        self.place_at_grid(torus_label, 'B3', scale_factor=0.6)
        
        # Context Group
        math_context_group = VGroup(torus_asset, torus_label)
        self.place_in_area(math_context_group, 'A4', 'F6', scale_factor=0.9)
        
        antipodal_1 = Dot(color="#FF0000")
        antipodal_2 = Dot(color="#FF0000")
        intersection_point = Dot(color="#00FF00")
        
        # Positioning antipodal points relative to torus
        antipodal_1.move_to(torus_asset.get_center() + np.array([0.5, 0.3, 0]))
        antipodal_2.move_to(torus_asset.get_center() - np.array([0.5, 0.3, 0]))
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(torus_asset), FadeIn(torus_label))
        self.lecture[0].set_color("#FFFF00")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(antipodal_1), FadeIn(antipodal_2))
        self.lecture[1].set_color("#FF0000")

        # === Animation for Lecture Line 3 ===
        self.play(FadeOut(antipodal_1), FadeOut(antipodal_2))
        self.lecture[2].set_color("#00FFFF")

        # === Animation for Lecture Line 4 ===
        # Square logic
        square = Square(side_length=0.5, color="#FFFF00")
        square.move_to(torus_asset.get_center())
        self.play(FadeIn(square))
        self.lecture[3].set_color("#FFFF00")

        # === Animation for Lecture Line 5 ===
        self.play(FadeIn(intersection_point))
        self.lecture[4].set_color("#00FF00")
        self.wait(1)
