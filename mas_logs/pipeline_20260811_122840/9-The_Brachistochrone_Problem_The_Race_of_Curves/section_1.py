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
            "What path connects two points in minimum time?",
            "Straight lines seem intuitive but aren't fastest.",
            "Gravity makes steep drops gain speed quickly."
        ]
        self.setup_layout("Introduction: The Fastest Path", lecture_lines)
        
        # Define mobjects
        dotA = Dot(color=WHITE)
        dotB = Dot(color=WHITE)
        labelA = Text("A", font_size=20, color=WHITE).scale(0.7)
        labelB = Text("B", font_size=20, color=WHITE).scale(0.7)
        
        # Asset loading
        particle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg", color="#FFFF00").scale(0.5)

        # === Animation for Lecture Line 1 ===
        # What path connects two points in minimum time?
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        
        self.place_at_grid(dotA, 'B2', scale_factor=0.8)
        self.place_at_grid(dotB, 'E5', scale_factor=0.8)
        
        labelA.next_to(dotA, UP, buff=0.1)
        labelB.next_to(dotB, RIGHT, buff=0.1)
        
        self.add(dotA, dotB, labelA, labelB)

        # === Animation for Lecture Line 2 ===
        # Straight lines seem intuitive but aren't fastest.
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        straight_line = Line(dotA.get_center(), dotB.get_center(), color="#00FF00")
        self.play(Create(straight_line))

        # === Animation for Lecture Line 3 ===
        # Gravity makes steep drops gain speed quickly.
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        particle.move_to(dotA.get_center())
        self.add(particle)
        self.play(MoveAlongPath(particle, straight_line), run_time=2, rate_func=linear)
        self.wait(1)
