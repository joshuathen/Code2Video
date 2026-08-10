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
        self.setup_layout("Prerequisite Review: The Static Slope", [
            "The secant line connects two distinct points.",
            "Its slope is the ratio, rise over run.",
            "Think of a hiker on a steep trail."
        ])
        
        # Create elements
        axes = Axes(x_range=[0, 4], y_range=[0, 4], axis_config={"include_tip": False}).scale(0.5)
        curve = axes.plot(lambda x: 0.2 * x**3, color=BLUE)
        
        pt_a = axes.coords_to_point(1, 0.2)
        pt_b = axes.coords_to_point(3, 1.8)
        
        dot_a = Dot(pt_a, color=WHITE)
        dot_b = Dot(pt_b, color=WHITE)
        label_a = Text("A", font_size=18).next_to(dot_a, DOWN)
        label_b = Text("B", font_size=18).next_to(dot_b, UP)
        secant = Line(pt_a, pt_b, color=YELLOW)
        
        hiker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hiker.svg").scale(0.3)
        slope_formula = MathTex(r"m = \frac{y_2 - y_1}{x_2 - x_1}", font_size=28)
        
        # Grouping for layout
        visual_group = VGroup(axes, curve, dot_a, dot_b, label_a, label_b, secant)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.place_in_area(visual_group, 'B4', 'E6', scale_factor=0.7)
        self.play(Create(axes), Create(curve), FadeIn(dot_a), FadeIn(dot_b), FadeIn(label_a), FadeIn(label_b))
        self.play(Create(secant), FadeIn(hiker.next_to(secant, UP)))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.place_at_grid(slope_formula, 'F1', scale_factor=0.9)
        self.play(Write(slope_formula))
        self.lecture[1].set_color(YELLOW)
        self.play(slope_formula.animate.set_color("#FFD700"))

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.lecture[2].set_color(YELLOW)
        self.wait(1)
