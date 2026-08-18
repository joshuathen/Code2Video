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
        lecture_lines = ["Standard convergence relies on Euclidean distance.", "Points close if distance is small.", "Like an ant approaching a crumb."]
        self.setup_layout("The Familiar World: Real Convergence", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        text_label = Text("Convergent sequence x_n -> L", font_size=24, color=WHITE)
        self.place_at_grid(text_label, "A5", scale_factor=0.9)
        self.play(FadeIn(text_label))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        number_line = NumberLine(x_range=[0, 1.2, 0.2], length=5, include_numbers=False)
        self.place_in_area(number_line, "D2", "D5", scale_factor=1.0)
        self.play(Create(number_line))
        
        # Crumb asset for points
        crumb = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/crumb.svg", color=BLUE)
        points = [1.0, 0.5, 0.25, 0.125]
        dots = VGroup(*[crumb.copy().scale(0.1).move_to(number_line.n2p(p)) for p in points])
        self.play(LaggedStart(*[FadeIn(dot) for dot in dots], lag_ratio=0.5))
        self.lecture[1].set_color("#0000FF")

        # === Animation for Lecture Line 3 ===
        ant = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ant.svg", fill_color=RED, stroke_color=RED, stroke_width=2)
        limit_point = number_line.n2p(0)
        ant.scale(0.15).move_to(limit_point)
        
        circle = Circle(radius=0.3, color=RED).move_to(limit_point)
        
        self.play(Create(circle), FadeIn(ant))
        self.play(circle.animate.scale(0.3), ant.animate.scale(0.3), run_time=1.5)
        
        checkmark = Tex(r"$\checkmark$", color=GREEN)
        self.place_at_grid(checkmark, "D1", scale_factor=0.7)
        self.play(Indicate(checkmark))
        
        self.lecture[2].set_color("#FF0000")
        self.wait(1)
