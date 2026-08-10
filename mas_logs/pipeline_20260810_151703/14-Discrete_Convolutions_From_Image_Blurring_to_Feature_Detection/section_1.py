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
        lecture_lines = ["Imagine a magnifying glass sliding over data.", "This filter captures local pixel patterns.", "It slides, computes, and shifts sequentially."]
        self.setup_layout("Intuitive Hook: The 'Moving Window' Concept", lecture_lines)
        
        # Pixel grid
        pixel_grid = VGroup(*[Square(side_length=0.7, color=BLUE_D, fill_opacity=0.3) for _ in range(36)])
        pixel_grid.arrange_in_grid(6, 6, buff=0.1)
        # Apply fix for Issue 21/23: Layout pixel_grid
        self.place_in_area(pixel_grid, 'B2', 'F6', scale_factor=0.65)
        
        # Filter window with Asset
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifying-glass.svg]
        mag_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifying-glass.svg", color=WHITE)
        window = VGroup(
            Rectangle(width=2.3, height=2.3, color=WHITE, stroke_width=4),
            mag_icon
        )
        # Apply fix for Issue 22: Layout window
        self.place_at_grid(window, 'D3', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(pixel_grid), FadeIn(window))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(window.animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        # Slide path
        path = VMobject()
        path.set_points_smoothly([self.grid['D3'], self.grid['D4'], self.grid['E4'], self.grid['E3']])
        self.play(MoveAlongPath(window, path), run_time=3)
        self.wait(1)
