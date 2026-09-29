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
            "Fluency requires moving between different representations.",
            "Algebra, verbal, and graphical forms work together.",
            "Visual-spatial reasoning reduces abstract cognitive load.",
            "Example: Pythagorean theorem water flow animation.",
            "Geometry transitions smoothly into symbolic algebra."
        ]
        self.setup_layout("Criterion 2: Multi-Representational Fluency", lecture_lines)

        # Pre-build objects
        graph_concept = Text("Concept A", color="#00FFFF")
        self.place_at_grid(graph_concept, 'C2', scale_factor=0.9)
        
        formula_eq = MathTex("a^2 + b^2 = c^2", color="#FF00FF")
        self.place_in_area(formula_eq, 'C3', 'D4', scale_factor=1.0)
        
        # Load asset
        water_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/water.svg")
        self.place_at_grid(water_icon, 'E3', scale_factor=0.5)
        
        arrow = Arrow(start=graph_concept.get_bottom(), end=formula_eq.get_top(), color="#FFFFFF")

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(graph_concept), run_time=1)
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(formula_eq), run_time=1)
        self.lecture[1].set_color("#FF00FF")

        # === Animation for Lecture Line 3 ===
        self.play(Create(arrow), run_time=1)
        self.lecture[2].set_color("#FFFFFF")

        # === Animation for Lecture Line 4 ===
        self.play(
            FadeIn(water_icon),
            Indicate(graph_concept, color="#FFFF00"), 
            Indicate(formula_eq, color="#FFFF00"),
            Indicate(water_icon, color="#FFFF00"),
            run_time=1.5
        )
        self.lecture[3].set_color("#FFFF00")

        # === Animation for Lecture Line 5 ===
        self.play(FadeOut(graph_concept), FadeOut(formula_eq), FadeOut(arrow), FadeOut(water_icon), run_time=1.5)
        self.lecture[4].set_color("#FFFFFF")
