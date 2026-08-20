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
        lecture_lines = ["Convergence can break unexpectedly.", "Guessing the root is chaotic.", "This forms a basin of attraction."]
        self.setup_layout("When Convergence Breaks: The Basin of Attraction", lecture_lines)
        
        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        graph = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": False}).scale(0.5)
        curve = graph.plot(lambda x: x**3 - x)
        self.place_in_area(graph, 'A1', 'C3', scale_factor=0.6)
        self.place_in_area(curve, 'A1', 'C3', scale_factor=0.6)
        self.place_at_grid(compass, 'A6', scale_factor=0.3)
        self.play(Create(graph), Create(curve), FadeIn(compass))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        points = VGroup(*[Dot(point=graph.c2p(x, 0), color=YELLOW) for x in np.linspace(-1.5, 1.5, 10)])
        self.play(FadeIn(points))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED)
        basin_rect = Rectangle(width=2, height=2, fill_opacity=0.3, color=PURPLE)
        self.place_at_grid(basin_rect, 'D4')
        self.place_at_grid(prism, 'F6', scale_factor=0.4)
        self.play(Create(basin_rect), FadeIn(prism))
        self.wait(1)
