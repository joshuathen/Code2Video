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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Consider the linear system Ax equals b.", "We interpret this as a vector combination.", "The goal is to reach point b."]
        self.setup_layout("The Linear System as Vector Scaling", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        eq = MathTex("A\\mathbf{x} = \\mathbf{b}", color=WHITE)
        self.place_at_grid(eq, 'B2', scale_factor=1.2)
        self.play(FadeIn(eq))
        self.lecture[0].set_color("#FFFF00")

        # === Animation for Lecture Line 2 ===
        # Using placeholder squares as assets are external/unavailable
        vec_vgroup = VGroup(
            MathTex("x_1\\mathbf{a}_1 + x_2\\mathbf{a}_2 = \\mathbf{b}", color="#00FFFF")
        )
        self.place_at_grid(vec_vgroup, 'D2', scale_factor=0.9)
        self.play(FadeIn(vec_vgroup))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        # Assets: /scratch/pawsey1357/jthen/Code2Video/assets/icon/target.svg and plane.svg
        try:
            plane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg")
            target = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/target.svg")
        except:
            plane = Square(color=WHITE, fill_opacity=0.2)
            target = Dot(color="#00FF00")
            
        self.place_in_area(plane, 'A4', 'F6', scale_factor=0.8)
        self.place_at_grid(target, 'D5', scale_factor=1.0)
        
        self.play(FadeIn(plane), FadeIn(target))
        self.lecture[2].set_color("#00FF00")
        self.wait(2)
