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
        # Initialize labels and lines
        title = "The Ant's Dilemma: Introduction to Dimensional Constraints"
        lines = [
            'Meet our ant, trapped inside a flat 2D circle.',
            'To the ant, this boundary is an impassable wall.',
            'But a new axis offers a path through height.',
            'The ant leaps over the once-solid barrier.',
            'It lands safely on the other side.'
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # 2D Plane representation (implicit in the background)
        circle = Circle(radius=1.5, color="#FF0000")
        self.place_in_area(circle, "B2", "E5", scale_factor=0.9)
        
        # Ant Asset
        ant = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/ant.svg")
        ant.set_color(WHITE) # Ensuring visibility
        # Place ant in center of circle area
        ant.scale(0.3).move_to(circle.get_center())
        
        self.play(Create(circle))
        self.play(FadeIn(ant))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Ant moves to edge (Right edge point)
        # Using C5 as a target grid point near the right edge
        self.play(ant.animate.move_to(self.grid["C5"]))
        self.play(Wiggle(ant))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Z-axis blue arrow tilted at 45 degrees
        z_axis = Arrow(start=ORIGIN, end=UP+RIGHT, color="#0000FF", buff=0)
        self.place_at_grid(z_axis, "A6", scale_factor=0.8)
        z_label = Text("Z-axis", font_size=16, color="#0000FF").next_to(z_axis, DOWN, buff=0.1)
        
        self.play(GrowArrow(z_axis), FadeIn(z_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        # Ant scales 1.5x and moves 'up and over' to B6
        self.play(
            ant.animate.scale(1.5).move_to(self.grid["B6"]),
            run_time=1.5
        )
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Ant scales back to 1.0x and settles on D6 (outside)
        self.play(
            ant.animate.scale(1/1.5).move_to(self.grid["D6"]),
            run_time=1.5
        )
        self.wait(2)
