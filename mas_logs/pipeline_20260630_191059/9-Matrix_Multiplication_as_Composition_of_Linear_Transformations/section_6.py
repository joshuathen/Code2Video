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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title_str = "Summary: The Computational Shortcut"
        lines = [
            "Matrix multiplication fuses multiple actions into one.",
            "This efficiency powers modern 3D computer graphics.",
            "Composition turns sequences into a single calculation."
        ]
        self.setup_layout(title_str, lines)

        # Define Colors
        color_fuses = "#87CEEB"  # Sky Blue
        color_graphics = "#98FB98"  # Pale Green
        color_composition = "#FFD700"  # Gold

        # Asset Paths
        robot_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/robot.svg"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color_fuses))

        # Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        matrix_a = Text("A", slant=ITALIC, color=color_fuses, font_size=36)
        matrix_b = Text("B", slant=ITALIC, color=color_fuses, font_size=36)
        matrix_c = Text("C", slant=ITALIC, color=color_fuses, font_size=36)
        
        self.place_at_grid(matrix_a, "B1")
        self.place_at_grid(matrix_b, "B2")
        self.place_at_grid(matrix_c, "B3")
        
        self.play(FadeIn(matrix_a), FadeIn(matrix_b), FadeIn(matrix_c))
        self.wait(1)

        # Merge into Matrix M using Text to avoid LaTeX dependency
        matrix_m = Text("M = C · B · A", slant=ITALIC, color=color_fuses, font_size=36)
        self.place_at_grid(matrix_m, "B5")
        
        self.play(
            ReplacementTransform(VGroup(matrix_a, matrix_b, matrix_c), matrix_m),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(color_graphics))

        # Coordinate Grid
        plane = NumberPlane(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            x_length=3, y_length=3,
            background_line_style={"stroke_opacity": 0.2}
        )
        self.place_in_area(plane, "C2", "E5")
        
        # Momo (Robot Asset)
        try:
            momo = SVGMobject(robot_path).set_color(color_graphics)
        except:
            momo = Circle(color=color_graphics, fill_opacity=0.5).add(Text("Momo", font_size=12))
        
        self.place_in_area(momo, "C2", "E5", scale_factor=0.5)
        
        self.play(Create(plane), FadeIn(momo))
        self.wait(0.5)

        # Fixed rate_func variables to standard Manim CE functions
        self.play(
            momo.animate.shift(UP * 0.5).rotate(2 * PI),
            run_time=2,
            rate_func=smooth
        )
        self.play(
            momo.animate.shift(DOWN * 0.5),
            run_time=0.5,
            rate_func=smooth # Fixed NameError: 'bounce' is not defined in Manim CE
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(color_composition))

        # Final Summary Text
        summary_text = Text("Matrix Multiplication = Composition", color=color_composition, font_size=24)
        self.place_in_area(summary_text, "F1", "F6")

        self.play(Write(summary_text))
        
        self.play(
            Indicate(matrix_m, color=color_composition),
            Indicate(summary_text, scale_factor=1.1)
        )
        
        self.wait(2)
