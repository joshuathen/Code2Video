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
            "Compare the areas of the substituted versus original shapes.",
            "x1 is the ratio of these specific parallelogram areas.",
            "Cramer's Rule interprets linear systems as area scaling."
        ]
        self.setup_layout("Visual Synthesis and Conclusion", lecture_lines)
        
        # Colors
        color_1 = "#FF9999" # Red-ish
        color_2 = "#99FF99" # Green-ish
        color_3 = "#9999FF" # Blue-ish

        # Setup parallelograms
        rect_original = Polygon(ORIGIN, RIGHT*2, RIGHT*2+UP*2, UP*2, color=WHITE)
        rect_sub = Polygon(ORIGIN, RIGHT*1.5, RIGHT*1.5+UP*2.5, UP*2.5, color=color_1)
        
        group = VGroup(rect_original, rect_sub)
        # Fix for issue 26/38: Move to area A4-C6
        self.place_in_area(group, 'A4', 'C6', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(group))
        self.lecture[0].set_color(color_1)

        # === Animation for Lecture Line 2 ===
        label_ratio = MathTex(r"x_1 = \frac{\text{Area}(A_1)}{\text{Area}(A)}", color=color_2)
        # Fix for issue 27/39: Position at D4
        self.place_at_grid(label_ratio, 'D4', scale_factor=0.9)
        
        self.play(Write(label_ratio))
        self.lecture[1].set_color(color_2)

        # === Animation for Lecture Line 3 ===
        label_cramer = Text("Cramer's Rule: Geometric Scaling", color=color_3, font_size=24)
        # Fix for issue 28/40: Position at E4
        self.place_at_grid(label_cramer, 'E4', scale_factor=0.8)
        
        self.play(FadeIn(label_cramer))
        self.lecture[2].set_color(color_3)
        
        self.wait(2)
        self.play(FadeOut(group), FadeOut(label_ratio), FadeOut(label_cramer))
