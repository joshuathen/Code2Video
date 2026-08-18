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
            "Limits describe behavior as x approaches a value.",
            "The function doesn't need to exist there.",
            "Consider f(x) equals x squared minus one over x minus one.",
            "A robot vacuum approaches a charger point.",
            "It converges without reaching the target."
        ]
        self.setup_layout("Intuitive Foundation: The Concept of 'Getting Closer'", lecture_lines)
        
        # Define elements
        axes = Axes(x_range=[-1, 3], y_range=[-1, 3], axis_config={"include_tip": False})
        func = lambda x: (x**2 - 1) / (x - 1) if x != 1 else 2
        graph = axes.plot(func, color=BLUE)
        hole = Dot(axes.c2p(1, 2), color=RED, radius=0.1)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg
        robot_vacuum = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg", color=YELLOW)
        
        target = Dot(axes.c2p(1, 2), color=YELLOW, radius=0.15) # Terminal anchor
        
        content_area = VGroup(axes, graph, hole, target)
        self.place_in_area(content_area, "B2", "F5", scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(axes), Create(graph))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(Create(hole))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        formula = MathTex("f(x) = \\frac{x^2-1}{x-1}").set_color(WHITE)
        self.place_at_grid(formula, "B1", scale_factor=0.6)
        self.play(Write(formula))

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        # Position using grid
        self.place_at_grid(robot_vacuum, "D4", scale_factor=0.5)
        self.add(robot_vacuum)
        
        # Animate robot vacuum path
        path = axes.plot(func, x_range=[0, 0.9])
        self.play(MoveAlongPath(robot_vacuum, path), run_time=3)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        self.play(FadeIn(target))
        self.wait(1)
