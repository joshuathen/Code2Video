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
            "The first derivative is a rate of change.",
            "It measures the slope of the tangent line.",
            "Example: Position becomes velocity. [Asset: f_prime_viz]"
        ]
        self.setup_layout("Prerequisite Review: The First Derivative", lecture_lines)
        
        # Assets
        car_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        
        # Elements
        graph = Axes(x_range=[0, 4], y_range=[0, 4], x_length=4, y_length=4, axis_config={"include_tip": False})
        curve = graph.plot(lambda x: 0.5 * x**2, color="#FFFFFF")
        dot = Dot(color="#00FFFF")
        dot.move_to(graph.c2p(1, 0.5))
        
        tangent_line = Line(start=ORIGIN, end=RIGHT*1.5, color="#FF00FF")
        slope_label = MathTex("f'(x)", color="#00FFFF")
        formula = MathTex("f'(x) = \\frac{dy}{dx}", color="#FFFF00")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.place_in_area(graph, 'B2', 'E5', scale_factor=0.6)
        self.place_at_grid(car_icon, 'B5', scale_factor=0.3)
        self.play(Create(graph), Create(curve), FadeIn(car_icon))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        # Position tangent line away from left text
        self.place_at_grid(tangent_line, 'C4', scale_factor=0.7)
        self.place_at_grid(slope_label, 'C3', scale_factor=0.75)
        self.play(Create(dot), Create(tangent_line), Write(slope_label))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        self.place_at_grid(formula, 'E4', scale_factor=0.8)
        self.place_at_grid(car_icon, 'E5', scale_factor=0.3)
        self.play(Write(formula), FadeIn(car_icon), MoveAlongPath(dot, curve), run_time=3)
        self.wait(1)
