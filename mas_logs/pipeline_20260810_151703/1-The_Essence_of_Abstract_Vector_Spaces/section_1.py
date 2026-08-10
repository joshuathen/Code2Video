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
            "Vectors start as arrows with magnitude and direction.",
            "We now focus on how they behave under operations.",
            "Think of these as abstract objects in a set.",
            "They follow specific rules, regardless of their appearance."
        ]
        self.setup_layout("From Concrete Arrows to Abstract Rules", lecture_lines)
        
        # Assets
        pencil = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pencil.svg")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")

        # === Animation for Lecture Line 1 ===
        vec = Arrow(start=ORIGIN, end=RIGHT*2 + UP*1, color="#FF5733")
        self.place_at_grid(vec, 'C3', scale_factor=1.0)
        self.place_at_grid(pencil, 'B2', scale_factor=0.5)
        self.play(Create(vec), FadeIn(pencil))
        self.lecture[0].set_color("#FF5733")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        dot = Dot(color="#33FF57")
        self.place_at_grid(dot, 'C3', scale_factor=0.5)
        self.play(Transform(vec, dot), FadeOut(pencil))
        self.lecture[1].set_color("#33FF57")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        label_v = MathTex(r"V", color=WHITE)
        label_x = MathTex(r"x", color="#33FF57")
        self.place_at_grid(label_v, 'B3', scale_factor=0.8)
        self.place_at_grid(label_x, 'D3', scale_factor=0.8)
        self.place_at_grid(ruler, 'D2', scale_factor=0.5)
        self.play(FadeIn(label_v), FadeIn(label_x), FadeIn(ruler))
        self.lecture[2].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        check = Tex(r"$\checkmark$", color=YELLOW)
        self.place_at_grid(check, 'E3', scale_factor=0.8)
        self.play(Write(check))
        self.lecture[3].set_color(YELLOW)
        self.wait(2)
