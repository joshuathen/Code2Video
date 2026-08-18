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
        self.setup_layout("Gradient Descent: The Mathematical Navigator", [
            "The gradient reveals the steepest slope.", 
            "We calculate it to find the descent path.", 
            "Moving opposite the gradient lowers the cost."
        ])
        
        # Elements
        func = lambda x: 0.1 * (x**2)
        axes = Axes(x_range=[-4, 4], y_range=[0, 2], axis_config={"include_tip": False})
        graph = axes.plot(func, color=BLUE)
        
        # Position graph area
        container = VGroup(axes, graph)
        self.place_in_area(container, 'A1', 'E6', scale_factor=0.7)
        
        dot = Dot(color=YELLOW).move_to(axes.c2p(3, func(3)))
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        grad_arrow = Arrow(start=dot.get_center(), end=dot.get_center() + RIGHT*0.8 + UP*0.5, color=RED)
        self.play(Create(dot), Create(grad_arrow))
        self.wait(3)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN)
        # Vector points opposite
        descent_arrow = Arrow(start=dot.get_center(), end=dot.get_center() + LEFT*0.8 + DOWN*0.5, color=GREEN)
        self.play(Transform(grad_arrow, descent_arrow))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(BLUE)
        
        # Animate descent
        self.play(MoveAlongPath(dot, Line(start=dot.get_center(), end=axes.c2p(0, 0))), run_time=3)
        self.play(FadeOut(grad_arrow))
        
        # Terminal anchor (moved to E1 as per recommendation)
        target = Circle(radius=0.2, color=ORANGE, fill_opacity=0.5)
        self.place_at_grid(target, 'E1', scale_factor=0.7)
        self.play(FadeIn(target))
        self.wait(2)
