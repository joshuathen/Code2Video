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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The Loss Function Landscape", [
            "Error is a valley in a mountain landscape.",
            "Lowest point represents perfect accuracy with zero error.",
            "Gradient shows the steepest direction toward the valley floor."
        ])

        # Assets
        mountain_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/mountain.svg"
        marble_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/marble.svg"

        # === Animation for Lecture Line 1 ===
        # Landscape
        mountain = SVGMobject(mountain_asset, color="#2266AA")
        self.place_in_area(mountain, 'B2', 'E6', scale_factor=0.5)
        
        self.play(FadeIn(mountain), self.lecture[0].animate.set_color("#2266AA"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Target Point
        target_dot = Dot(color=WHITE)
        target_label = Text("Target", font_size=20, color=WHITE)
        
        self.place_at_grid(target_dot, 'D5', scale_factor=1.0)
        self.place_at_grid(target_label, 'C2', scale_factor=0.7)
        
        self.play(FadeIn(target_dot), Write(target_label), self.lecture[1].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Marble (Hiker)
        marble = SVGMobject(marble_asset, color="#FFFFFF")
        gradient_arrow = Arrow(start=UP*0.2, end=DOWN*0.2, color="#FFCC00")
        
        self.place_at_grid(marble, 'B3', scale_factor=0.4)
        gradient_arrow.next_to(marble, DOWN, buff=0.1)
        
        self.play(FadeIn(marble), GrowArrow(gradient_arrow), self.lecture[2].animate.set_color("#FFCC00"))
        self.play(FadeOut(gradient_arrow))
        self.wait(1)
