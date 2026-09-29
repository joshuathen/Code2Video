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
        self.setup_layout("Prerequisite Review: Vectors and Areas", [
            "A vector is a directed line segment in 3D.", 
            "Vectors u and v span a 2D parallelogram.", 
            "This area reflects their geometric interaction magnitude."
        ])
        
        # Vectors u and v
        u = Vector([1, 1.5, 0], color="#FF00FF").shift(DOWN + LEFT)
        v = Vector([2, 0.5, 0], color="#00FFFF").shift(DOWN + LEFT)
        
        u_label = MathTex("u", color="#FF00FF")
        v_label = MathTex("v", color="#00FFFF")
        self.place_at_grid(u_label, 'B4', scale_factor=0.9)
        self.place_at_grid(v_label, 'C6', scale_factor=0.9)
        
        # Use asset for parallelogram
        parallelogram_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/parallelogram.svg", 
                                        fill_color=BLUE, fill_opacity=0.3, stroke_color=WHITE)
        parallelogram_group = VGroup(parallelogram_asset)
        self.place_in_area(parallelogram_group, 'B3', 'C6', scale_factor=1.2)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF00FF")
        self.play(Create(u), Write(u_label), Create(v), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        self.play(Create(parallelogram_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        area_text = Text("Area", font_size=24, color=YELLOW)
        self.place_at_grid(area_text, 'D3', scale_factor=1.0)
        self.play(Write(area_text))
        self.wait(2)
