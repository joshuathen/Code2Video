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
        lecture_lines = [
            "The cross product u × v produces a new vector.",
            "It is always perpendicular to the original plane.",
            "Magnitude equals the spanned parallelogram area."
        ]
        self.setup_layout("Defining the Cross Product: Direction and Magnitude", lecture_lines)
        
        # Objects
        # Using 2D representations because of the 2D grid/setup limitations in TeachingScene base
        axes = Axes(x_range=[-3,3], y_range=[-3,3], axis_config={"include_tip": True})
        u = Vector([2, 0.5], color=BLUE)
        v = Vector([0.5, 1.5], color=YELLOW)
        w = Vector([0, 2], color=RED)
        # Fix: Convert list-based vertices to np.array to ensure consistent shape (N, 3)
        plane = Polygon(np.array([0, 0, 0]), np.array([2, 0.5, 0]), np.array([2.5, 2, 0]), np.array([0.5, 1.5, 0]), color=WHITE, fill_opacity=0.3)
        
        # Load asset
        hand = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hand.svg", color=WHITE)
        
        right_side_group = VGroup(axes, u, v, w, plane)
        self.place_in_area(right_side_group, 'B2', 'E5', scale_factor=1.2)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Create(u), Create(v))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(RED))
        self.place_at_grid(hand, 'B5', scale_factor=0.3)
        self.play(FadeIn(hand))
        self.play(Create(plane), GrowArrow(w))
        self.play(FadeOut(hand))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(Indicate(plane))
        self.wait(2)
