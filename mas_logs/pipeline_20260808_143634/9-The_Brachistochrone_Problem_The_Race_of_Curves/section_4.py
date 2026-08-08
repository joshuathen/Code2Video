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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Solution: The Cycloid", [
            "The optimal curve is a cycloid.", 
            "It traces a rolling circle's rim point.", 
            "Cycloids balance path length and speed gains."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Using the asset
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg")
        self.place_at_grid(circle, 'B3', scale_factor=0.75)
        
        circle_label = Text("Rolling Circle", font_size=24, color=WHITE)
        self.place_at_grid(circle_label, 'B4', scale_factor=0.5)
        
        # Rim dot
        dot = Dot(color=RED)
        dot.move_to(circle.get_bottom())
        
        # TracedPath for the cycloid
        path = TracedPath(dot.get_center, stroke_color=BLUE, stroke_width=3)
        self.add(path)
        
        self.play(self.lecture[1].animate.set_color(GREEN))
        
        # Rolling movement
        # Adjust dot/circle movement to account for cycloid path
        # Simplification: animate shift directly
        self.play(
            circle.animate.shift(RIGHT * 3),
            dot.animate.shift(RIGHT * 3),
            run_time=2,
            rate_func=linear
        )
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        self.wait(2)
