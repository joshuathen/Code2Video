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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "The Sellmeier equation describes material response.",
            "Refractive index depends on wavelength constants.",
            "Values peak near material absorption bands.",
            "Indices change rapidly near resonance.",
            "Prisms reveal this through color spreading."
        ]
        
        self.setup_layout("Visualizing the Mathematics: The Sellmeier Equation", lecture_lines)
        
        # Define equations
        sellmeier = MathTex(r"n^2(\lambda) = 1 + \sum_i \frac{B_i \lambda^2}{\lambda^2 - C_i}")
        
        # Load asset
        prism_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")
        
        # === Animation for Lecture Line 1 ===
        # The Sellmeier equation describes material response.
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        # Placing sellmeier in area as per Critic requirement 30
        self.place_in_area(sellmeier, 'A4', 'C6', scale_factor=0.9)
        self.play(Write(sellmeier))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Refractive index depends on wavelength constants.
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        terms = sellmeier.get_parts_by_tex("B_i")
        terms2 = sellmeier.get_parts_by_tex("C_i")
        self.play(Indicate(terms), Indicate(terms2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Values peak near material absorption bands.
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        # Highlight resonance with asset - requirement 19
        resonance_highlight = prism_asset.copy()
        self.place_in_area(resonance_highlight, 'A4', 'B6', scale_factor=0.8) # Requirement 32
        self.play(Create(resonance_highlight))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Indices change rapidly near resonance.
        self.play(self.lecture[3].animate.set_color("#FF0000"))
        arrow = Arrow(start=self.grid['D4'], end=self.grid['B4'], color=WHITE)
        self.play(GrowArrow(arrow))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Prisms reveal this through color spreading.
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        prism = prism_asset.copy()
        self.place_at_grid(prism, 'E5', scale_factor=0.7) # Requirement 31
        self.play(Create(prism))
        self.wait(2)
