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
        self.setup_layout("The Counter-Intuitive 'Empty' Center", [
            "Imagine a high-dimensional orange.",
            "Almost all mass lives in the crust.",
            "The core is effectively empty space."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show 2D circle with full shaded area
        circle = Circle(radius=1.5, color="#0000FF", fill_opacity=0.6)
        self.place_in_area(circle, 'A4', 'C6', scale_factor=0.7)
        self.play(FadeIn(circle))
        self.lecture[0].set_color("#0000FF")

        # === Animation for Lecture Line 2 ===
        # Load asset and show as representative of the sphere
        orange_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/orange.svg")
        sphere_shell = Annulus(inner_radius=1.0, outer_radius=1.5, color="#FF4500", fill_opacity=0.5)
        
        # Combine asset with the sphere representation
        combined_obj = VGroup(orange_asset, sphere_shell)
        
        self.place_in_area(combined_obj, 'D4', 'F6', scale_factor=0.8)
        self.play(FadeOut(circle), FadeIn(combined_obj))
        self.lecture[1].set_color("#FF4500")

        # === Animation for Lecture Line 3 ===
        # Highlight the thin shell volume of an N-sphere
        shell_highlight = Annulus(inner_radius=1.3, outer_radius=1.5, color="#FFD700", fill_opacity=0.8)
        
        self.place_in_area(shell_highlight, 'D4', 'F6', scale_factor=0.82)
        self.play(ReplacementTransform(sphere_shell, shell_highlight))
        self.lecture[2].set_color("#FFD700")
        self.wait(2)
