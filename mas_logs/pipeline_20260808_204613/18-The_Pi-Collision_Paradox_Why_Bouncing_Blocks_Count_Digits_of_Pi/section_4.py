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
        self.setup_layout("The Connection to Pi", [
            "Mass ratio controls the wedge angle.",
            "More mass means more reflections.",
            "The count reveals digits of Pi.",
            "100 leads to 31 collisions.",
            "10,000 leads to 314 collisions."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Mass ratio formula
        ratio_formula = MathTex(r"M/m \to 100^n").set_color("#FFFFFF")
        self.place_at_grid(ratio_formula, 'B2', scale_factor=1.2)
        self.play(FadeIn(ratio_formula))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Reflections in wedge
        wedge = VGroup(
            Line(ORIGIN, 3*RIGHT),
            Line(ORIGIN, 3*(RIGHT*np.cos(0.2) + UP*np.sin(0.2)))
        ).set_color("#FFFF00")
        self.place_in_area(wedge, 'C2', 'D5', scale_factor=0.8)
        self.play(Create(wedge))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # Digits of Pi
        pi_digits = Text("3.14159...").set_color("#00FF00")
        self.place_at_grid(pi_digits, 'E2', scale_factor=1.0)
        self.play(Write(pi_digits))
        self.lecture[2].set_color("#00FF00")

        # === Animation for Lecture Line 4 ===
        # 100 leads to 31 collisions
        block100 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg").set_color("#FF0000")
        label100 = Text("100 = 31").set_color("#FF0000")
        container100 = VGroup(block100, label100).arrange(DOWN)
        self.place_at_grid(container100, 'B5', scale_factor=0.6)
        self.play(FadeIn(container100))
        self.lecture[3].set_color("#FF0000")

        # === Animation for Lecture Line 5 ===
        # 10,000 leads to 314 collisions
        block10k = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg").set_color("#FF0000")
        label10k = Text("10,000 = 314").set_color("#FF0000")
        container10k = VGroup(block10k, label10k).arrange(DOWN)
        self.place_at_grid(container10k, 'E5', scale_factor=0.6)
        self.play(FadeIn(container10k))
        self.lecture[4].set_color("#FF0000")
        
        self.wait(2)
