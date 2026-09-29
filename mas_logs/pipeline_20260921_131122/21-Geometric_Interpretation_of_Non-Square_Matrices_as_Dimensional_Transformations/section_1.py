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
        self.setup_layout("Prerequisite Refresher: The Linear Map", [
            "Matrices are functions mapping vectors between spaces.",
            "Square matrices stay in the same dimension.",
            "Non-square matrices change the space dimension."
        ])
        
        # Assets
        # Note: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg] was requested but 
        # is just a placeholder icon. I will use SVG图标 as a stand-in or just rely on the vectors
        # if the asset doesn't provide visual value.
        
        # Objects
        v = Arrow(start=ORIGIN, end=RIGHT*1.5, color=WHITE).set_stroke(width=4)
        v_label = Text("v", font_size=24, color=WHITE).next_to(v.get_end(), UP, buff=0.1)
        v_group = VGroup(v, v_label)
        
        av = Arrow(start=ORIGIN, end=UP*1.2+RIGHT*0.5, color="#ADD8E6").set_stroke(width=4)
        av_label = Text("Av", font_size=24, color="#ADD8E6").next_to(av.get_end(), RIGHT, buff=0.1)
        av_group = VGroup(av, av_label)
        
        prop_text = Text("L(u+v) = L(u)+L(v)", font_size=20, color="#90EE90")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        self.place_in_area(v_group, 'B2', 'C3', scale_factor=0.95)
        self.play(FadeIn(v_group))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#ADD8E6")
        self.play(ReplacementTransform(v_group.copy(), av_group))
        self.play(FadeIn(av_group))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#90EE90")
        self.place_at_grid(prop_text, 'D4', scale_factor=0.9)
        self.play(Write(prop_text))
        self.wait(2)
