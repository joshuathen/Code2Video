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
        self.setup_layout("The Inscribed Square Problem", [
            "Can we find four points forming a square?",
            "Any simple closed curve allows this configuration.",
            "Visualize a messy loop of string.",
            "Place a square stencil on the loop.",
            "All four corners touch the string."
        ])
        
        # Assets
        string_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/string.svg", color=WHITE)
        stencil_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/stencil.svg")
        
        # Initial positions
        self.place_in_area(string_svg, "B4", "E6")
        self.add(string_svg)
        
        # Ensure the string_svg has points before calling geometric methods
        string_path = (string_svg if isinstance(string_svg, VMobject) and string_svg.has_points() else string_svg[0])
        
        quad = Polygon(
            string_path.point_from_proportion(0),
            string_path.point_from_proportion(0.25),
            string_path.point_from_proportion(0.5),
            string_path.point_from_proportion(0.75),
            color=YELLOW
        )

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(Create(quad))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(string_svg.animate.rotate(0.2))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        self.place_at_grid(stencil_svg, "C4", scale_factor=0.5)
        self.play(FadeIn(stencil_svg))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(YELLOW))
        # Highlight corners
        self.play(Flash(quad.get_vertices()[0]), Flash(quad.get_vertices()[1]), 
                  Flash(quad.get_vertices()[2]), Flash(quad.get_vertices()[3]))
        
        self.wait(2)
