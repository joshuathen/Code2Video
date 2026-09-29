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
        lecture_lines = [
            "Linear transformations stretch and rotate space.",
            "Most vectors change direction during transformation.",
            "Some special vectors stay on their span.",
            "These special vectors are our focus.",
            "They reveal the transformation's underlying geometry."
        ]
        self.setup_layout("The Hook: Stretching Space", lecture_lines)
        
        # Assets
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        
        # Elements
        grid_display = self.place_in_area(grid_asset.copy(), 'A1', 'F6', scale_factor=0.8)
        vec_1 = Arrow(start=ORIGIN, end=RIGHT*1.5, color=ORANGE)
        vec_2 = Arrow(start=ORIGIN, end=UP*1.5, color=ORANGE)
        span_line = Line(start=LEFT*2, end=RIGHT*2, color=GREEN)
        special_vec = Arrow(start=ORIGIN, end=RIGHT*1.5+UP*0.5, color=YELLOW)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        self.play(FadeIn(grid_display))
        self.play(grid_display.animate.apply_matrix([[1.5, 0.5], [0, 1]]), run_time=2)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF4500")
        self.play(FadeIn(vec_1), FadeIn(vec_2))
        self.play(vec_1.animate.rotate(0.5), vec_2.animate.rotate(0.5))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        self.play(Create(span_line))
        self.play(FadeOut(vec_1), FadeOut(vec_2))
        self.play(GrowArrow(special_vec))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        self.play(Indicate(special_vec))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        grid_overlay = self.place_in_area(grid_asset.copy(), 'A1', 'F6', scale_factor=0.8)
        self.play(FadeIn(grid_overlay), FadeIn(special_vec))
        self.wait(2)
