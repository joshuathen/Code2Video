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
            "Consider the integral of sine to the power n.",
            "As n increases, the area shrinks significantly.",
            "The reduction formula links n to n-2."
        ]
        self.setup_layout("Prerequisites: The Power of Sine Powers", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Color: #FFFFFF
        integral_formula = MathTex(r"I_n = \int_0^{\pi/2} \sin^n(x) \, dx", color=WHITE)
        self.place_in_area(integral_formula, 'B3', 'C4', scale_factor=0.7)
        self.play(Write(integral_formula))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        # Color: #FFCC00 (graph), #00FFCC (area), #FF6600 (shrinking)
        axes = Axes(x_range=[0, PI/2 + 0.1, 0.5], y_range=[0, 1.1, 0.5], axis_config={"include_tip": False})
        axes.scale(0.8).move_to(self.grid["E4"])
        
        # Including Asset (Icon)
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        self.place_at_grid(icon, "A6", scale_factor=0.5)
        self.add(icon)
        
        graph = axes.plot(lambda x: np.sin(x)**1, color="#FFCC00")
        area = axes.get_area(graph, x_range=[0, PI/2], color="#00FFCC", opacity=0.5)
        
        self.play(Create(axes), Create(graph), FadeIn(area))
        
        # Animation of shrinking
        for n in range(2, 6):
            new_graph = axes.plot(lambda x: np.sin(x)**n, color="#FF6600")
            new_area = axes.get_area(new_graph, x_range=[0, PI/2], color="#FF6600", opacity=0.5)
            self.play(Transform(graph, new_graph), Transform(area, new_area))
        
        self.lecture[1].set_color("#FF6600")

        # === Animation for Lecture Line 3 ===
        # Color: #FFFFFF
        reduction = MathTex(r"I_n = \frac{n-1}{n} I_{n-2}", color=WHITE)
        self.place_at_grid(reduction, 'C5', scale_factor=0.8)
        self.play(Write(reduction))
        self.play(Indicate(reduction, color=WHITE))
        self.lecture[2].set_color(WHITE)
        
        self.wait(2)
