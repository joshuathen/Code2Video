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
        self.setup_layout("The Root Problem: Filtering Coefficients", [
            "Challenge: Summing every k-th coefficient of P(x).",
            "Use roots of unity as a filter.",
            "Roots vanish when powers aren't multiples of k."
        ])
        
        # Coefficients representation
        coeffs = VGroup(*[Text(f"a_{i}", font_size=24) for i in range(8)])
        coeffs.arrange(RIGHT, buff=0.3)
        # Applying fixes from VideoCritic (31): Use C3, scale 0.8
        self.place_at_grid(coeffs, "C3", scale_factor=0.8)
        
        # Asset for Lecture Line 3
        filter_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/filter.svg")
        self.place_at_grid(filter_icon, "D3", scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        # Display coefficient sequence on screen. (#FFFFFF)
        self.play(FadeIn(coeffs))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Highlight every k-th coefficient using flashing effect. (#FF5733)
        self.lecture[1].set_color("#FF5733")
        k = 3
        highlights = VGroup(*[SurroundingRectangle(coeffs[i], color="#FF5733") for i in range(0, 8, k)])
        self.play(Create(highlights))
        self.play(Flash(highlights, color="#FF5733", flash_radius=0.3))

        # === Animation for Lecture Line 3 ===
        # Show zeroing out of non-matching coefficients. (#FF33A8)
        self.lecture[2].set_color("#FF33A8")
        zeroing = VGroup()
        for i in range(8):
            if i % k != 0:
                zeroing.add(coeffs[i])
        
        self.play(FadeIn(filter_icon))
        self.play(FadeOut(zeroing, shift=DOWN*0.5))
        self.play(FadeOut(highlights))
