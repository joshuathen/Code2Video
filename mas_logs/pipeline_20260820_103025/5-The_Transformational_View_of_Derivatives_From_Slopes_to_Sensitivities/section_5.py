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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Synthesis and Summary", ["Derivatives transform states into rates of change.", "Local linear approximation provides transformational power.", "Summarize the path: Function, Derivative, Context."])
        
        # Setup visual elements
        table = VGroup(
            Text("Function (State)", font_size=24, color=BLUE),
            Text("Derivative (Change)", font_size=24, color=GREEN),
            Text("Context (Trend)", font_size=24, color=RED)
        ).arrange(DOWN, aligned_edge=LEFT)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        # Load asset
        path_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", color="#FFFFFF")
        self.place_at_grid(path_icon, "B4", scale_factor=0.7)
        self.play(FadeIn(path_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        self.place_in_area(table, "A3", "C5", scale_factor=0.7)
        self.play(Write(table))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF0000")
        formula = MathTex(r"f'(x) = \lim_{h \to 0} \frac{f(x+h)-f(x)}{h}", color="#FF0000")
        self.place_at_grid(formula, "E5", scale_factor=0.8)
        self.play(formula.animate.scale(1.2), run_time=0.5)
        self.play(formula.animate.scale(1/1.2), run_time=0.5)
        self.wait(2)
