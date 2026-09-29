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
        lecture_lines = ["Each finite iteration remains one-dimensional.", "The limit reaches every point inside.", "The mapping is continuous, not injective.", "The ant traverses all coordinates.", "Infinite complexity arises from simple rules."]
        self.setup_layout("Finite Rules, Infinite Results", lecture_lines)

        # Animation state
        path_color = YELLOW
        subdiv_color = GREEN
        initial_color = WHITE

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(initial_color)
        square = Square(side_length=2.0, color=initial_color)
        self.place_in_area(square, 'C4', 'E5', scale_factor=0.6)
        self.play(Create(square))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(subdiv_color)
        subdivs = VGroup(*[Square(side_length=1.0, color=subdiv_color) for _ in range(4)])
        self.place_at_grid(subdivs[0], 'C4', scale_factor=0.5)
        self.place_at_grid(subdivs[1], 'C5', scale_factor=0.5)
        self.place_at_grid(subdivs[2], 'D4', scale_factor=0.5)
        self.place_at_grid(subdivs[3], 'D5', scale_factor=0.5)
        self.play(ReplacementTransform(square, subdivs))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        path = Line(subdivs[0].get_center(), subdivs[1].get_center(), color=YELLOW)
        path.add_line_to(subdivs[3].get_center())
        path.add_line_to(subdivs[2].get_center())
        self.play(Create(path))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF00FF")
        ant = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ant.svg", color=RED)
        self.place_at_grid(ant, 'C4', scale_factor=0.3)
        self.play(FadeIn(ant))
        self.play(MoveAlongPath(ant, path))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        final_dot = Dot(self.grid["D5"], color=GREEN_B)
        self.play(FadeIn(final_dot))
        self.wait(1)
