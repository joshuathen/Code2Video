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
        lecture_lines = ["Calculate score: Q dot K.", "Visualize as a heat map.", "Normalize scores using Softmax.", "Scores now sum to one.", "Words get a weighted importance."]
        self.setup_layout("The Math: Dot-Product and Softmax", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        eq = MathTex(r"Q \cdot K^T", color=WHITE)
        self.place_in_area(eq, 'A2', 'B4', scale_factor=1.2)
        self.play(Write(eq))
        self.lecture[0].set_color("#FFFF00")

        # === Animation for Lecture Line 2 ===
        # Using SVG asset
        matrix_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/matrix.svg", color=WHITE)
        self.place_at_grid(matrix_icon, 'C3', scale_factor=0.5)
        heatmap = VGroup(*[Square(side_length=0.7, fill_opacity=0.6, fill_color=interpolate_color(BLUE, RED, i/8), stroke_width=0) for i in range(9)]).arrange_in_grid(3, 3)
        self.place_at_grid(heatmap, 'D2', scale_factor=0.7)
        self.play(FadeIn(heatmap), FadeIn(matrix_icon))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        softmax_text = MathTex(r"\text{Softmax}", color=WHITE)
        self.place_at_grid(softmax_text, 'B5', scale_factor=1.2)
        self.play(Write(softmax_text))
        self.lecture[2].set_color("#FFFF00")

        # === Animation for Lecture Line 4 ===
        matrix_result = VGroup(*[Square(side_length=0.7, fill_opacity=0.8, fill_color="#7FFF00", stroke_width=1) for i in range(9)]).arrange_in_grid(3, 3)
        heatmap_overlay = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/heat-map.svg")
        self.place_at_grid(matrix_result, 'D4', scale_factor=0.7)
        self.place_at_grid(heatmap_overlay, 'D4', scale_factor=0.5)
        self.play(Transform(heatmap, matrix_result), FadeIn(heatmap_overlay))
        self.lecture[3].set_color("#7FFF00")

        # === Animation for Lecture Line 5 ===
        final_label = Text("Attention Weights", font_size=24, color=WHITE)
        self.place_at_grid(final_label, 'B5', scale_factor=0.9)
        self.play(Write(final_label))
        self.lecture[4].set_color("#7FFF00")
        
        self.wait(2)
