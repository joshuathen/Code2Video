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
        self.setup_layout("Critical Edge Cases: Determinant = 0", [
            "What if the determinant is zero?",
            "Space collapses into a line.",
            "Information is lost forever."
        ])
        
        # Elements
        square = Square(side_length=2, color=BLUE).set_fill(BLUE, opacity=0.5)
        self.place_at_grid(square, 'C5', scale_factor=0.6)
        
        # Using asset per issue 24 requirements
        collapsed_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/line.svg")
        self.place_at_grid(collapsed_icon, 'C5', scale_factor=0.6)
        collapsed_icon.set_opacity(0)
        
        area_label = Text("Area = 0", color=RED, font_size=24)
        self.place_at_grid(area_label, 'E5', scale_factor=0.8)
        area_label.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(square))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE),
                  self.lecture[1].animate.set_color(YELLOW))
        self.play(
            square.animate.stretch(0, 0),
            FadeOut(square),
            FadeIn(collapsed_icon),
            run_time=2
        )
        self.play(collapsed_icon.animate.set_color("#808080"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE),
                  self.lecture[2].animate.set_color(YELLOW))
        self.play(FadeIn(area_label))
        self.wait(2)
