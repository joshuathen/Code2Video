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
        lecture_lines = ["Scalars scale vector magnitude.", "Multiplying by two doubles reach.", "Negative values reverse the direction."]
        self.setup_layout("Scalar Multiplication", lecture_lines)
        
        # Axes
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_tip": True}).scale(0.5)
        self.place_in_area(axes, "A3", "F6", scale_factor=0.6)
        self.add(axes)

        # Assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        
        # Initial Vector
        v1_coords = [1, 2]
        v1 = Vector(axes.c2p(*v1_coords), color="#E74C3C")
        self.place_at_grid(v1, "D4", scale_factor=0.7) # Adjusted from D3 based on grid usage
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#E74C3C")
        ruler_1 = ruler.copy()
        self.place_at_grid(ruler_1, "B4", scale_factor=0.3)
        self.play(Create(v1), FadeIn(ruler_1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#E74C3C")
        v2_coords = [2, 4]
        v2 = Vector(axes.c2p(*v2_coords), color="#E74C3C")
        self.place_at_grid(v2, "D4", scale_factor=0.7)
        self.play(Transform(v1, v2), FadeOut(ruler_1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#9B59B6")
        v3_coords = [-1, -2]
        v3 = Vector(axes.c2p(*v3_coords), color="#9B59B6")
        self.place_at_grid(v3, "D4", scale_factor=0.7)
        
        ruler_2 = ruler.copy()
        self.place_at_grid(ruler_2, "E4", scale_factor=0.3)
        
        self.play(Transform(v1, v3), FadeIn(ruler_2))
        self.wait(1)
