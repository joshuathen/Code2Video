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
        lecture_lines = ["Connect the head to the tail.", "This adds two vectors together.", "The result is a new vector."]
        self.setup_layout("Vector Addition: The Head-to-Tail Rule", lecture_lines)
        
        # Pencil and Ruler icons (assets)
        pencil = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pencil.svg")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        
        u = Arrow(start=ORIGIN, end=RIGHT*1.5 + UP*0.5, color="#00FFFF")
        v = Arrow(start=ORIGIN, end=RIGHT*0.5 + UP*1.5, color="#FF00FF")
        
        u_label = MathTex(r"\vec{u}", color="#00FFFF")
        v_label = MathTex(r"\vec{v}", color="#FF00FF")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.place_at_grid(pencil, "A3", scale_factor=0.3)
        self.play(FadeIn(pencil))
        self.place_at_grid(u, "C4")
        self.place_at_grid(u_label, "B3", scale_factor=0.6)
        self.play(Create(u), Write(u_label))
        self.play(FadeOut(pencil))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        v.shift(u.get_end())
        self.place_at_grid(v_label, "B5", scale_factor=0.6)
        self.play(Create(v), Write(v_label))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # resultant
        resultant = Arrow(start=ORIGIN, end=u.get_vector() + v.get_vector(), color=WHITE)
        resultant.shift(u.get_start())
        resultant_label = MathTex(r"\vec{u} + \vec{v}", color=WHITE)
        
        self.place_in_area(VGroup(u, v, resultant), "C4", "E6", scale_factor=0.8)
        self.place_at_grid(ruler, "F6", scale_factor=0.4)
        self.place_at_grid(resultant_label, "C5", scale_factor=0.7)
        
        self.play(Create(resultant), Write(resultant_label), FadeIn(ruler))
        self.wait(2)
