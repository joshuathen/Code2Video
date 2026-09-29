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
        lecture_lines = [
            "Define the sine integral: I_n from zero to pi/2.",
            "Observe the area shrinking as n increases.",
            "Use the reduction formula: I_n equals (n-1)/n I_{n-2}.",
            "Compare even and odd powers for insight.",
            "This formula reveals pi's structure through sine."
        ]
        self.setup_layout("Prerequisite: The Reduction Formula for Sine", lecture_lines)
        
        # Load Assets
        graph_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg")
        protractor_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        
        # Define equations
        formula1 = MathTex(r"I_n = \int_{0}^{\pi/2} \sin^n(x) \, dx", color="#33FF57")
        formula2 = MathTex(r"I_n = \frac{n-1}{n} I_{n-2}", color="#33FF57")
        
        # === Animation for Lecture Line 1 ===
        # Using recommendation from issue 37
        self.place_in_area(formula1, 'B3', 'C5', scale_factor=0.9)
        self.place_at_grid(graph_icon, 'A5', scale_factor=0.4)
        self.play(Write(formula1), FadeIn(graph_icon))
        self.lecture[0].set_color("#33FF57")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        axes = Axes(x_length=3, y_length=2, x_range=[0, PI/2, 0.5], y_range=[0, 1, 0.5], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: np.sin(x)**2, color=BLUE)
        area = axes.get_area(curve, [0, PI/2], color=BLUE, opacity=0.3)
        vis_group = VGroup(axes, curve, area)
        self.place_in_area(vis_group, "D3", "E5", scale_factor=0.8)
        self.play(Create(axes), Create(curve), FadeIn(area))
        self.lecture[1].set_color("#FFFF33")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(FadeOut(vis_group), FadeOut(graph_icon))
        self.place_in_area(formula2, 'D3', 'E5', scale_factor=0.9)
        self.play(Write(formula2))
        self.lecture[2].set_color("#33FF57")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.place_at_grid(protractor_icon, 'B5', scale_factor=0.5)
        self.play(FadeIn(protractor_icon))
        highlight = SurroundingRectangle(formula2[0][2:5], color="#FFFF33")
        self.play(Create(highlight))
        self.lecture[3].set_color("#FFFF33")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(FadeOut(highlight), FadeOut(protractor_icon))
        self.lecture[4].set_color("#33FF57")
        self.wait(2)
