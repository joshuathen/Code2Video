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
            "Measure change over an interval with secant lines.",
            "Visualize squirrel speed: distance divided by time.",
            "Average rate of change forms the foundational baseline."
        ]
        self.setup_layout("Prerequisite: The Concept of Rate of Change", lecture_lines)
        
        # Define elements
        dot_a = Dot(color="#FFD700")
        dot_b = Dot(color="#FFD700")
        squirrel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/squirrel.svg")
        
        # Using mandated grid positioning
        self.place_at_grid(dot_a, 'B5', scale_factor=0.6)
        self.place_at_grid(dot_b, 'E5', scale_factor=0.6)
        self.place_at_grid(squirrel, 'B4', scale_factor=0.3)
        
        secant = Line(dot_a.get_center(), dot_b.get_center(), color="#00FFFF")

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(dot_a), FadeIn(dot_b), FadeIn(squirrel))
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(Create(secant))
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate shrinking gap
        new_b_pos = self.grid['D5']
        
        # Squirrel must follow B
        squirrel_tracker = Mobject()
        squirrel_tracker.add_updater(lambda m: m.move_to(dot_b.get_center() + np.array([-0.3, 0.3, 0])))
        self.add(squirrel_tracker)
        
        self.play(
            dot_b.animate.move_to(new_b_pos),
            UpdateFromAlphaFunc(secant, lambda m, a: m.put_start_and_end_on(dot_a.get_center(), dot_b.get_center()))
        )
        self.play(self.lecture[2].animate.set_color("#FF4500"))
        self.wait(1)
