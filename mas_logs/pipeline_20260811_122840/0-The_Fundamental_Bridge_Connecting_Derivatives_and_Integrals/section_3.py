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
        lecture_lines = [
            "Integrals accumulate many tiny slices into one sum.",
            "Think of filling a tank over time.",
            "Adding these slices gives the total amount."
        ]
        self.setup_layout("The Integral as 'Summation'", lecture_lines)

        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": True}).scale(0.6)
        # Apply fix for Issue 29: adjust axes area
        self.place_in_area(axes, 'B3', 'E5', scale_factor=0.75)
        self.add(axes)

        curve = axes.plot(lambda x: 0.5 * x**2, color=WHITE)
        self.add(curve)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        rects = VGroup(*[
            Rectangle(height=axes.c2p(0, 0.5 * x**2)[1] - axes.c2p(0, 0)[1], width=0.4, color=BLUE, fill_opacity=0.3)
            .move_to(axes.c2p(x, 0.25 * x**2), aligned_edge=DOWN)
            for x in np.arange(0.5, 3.5, 0.5)
        ])
        self.play(Create(rects))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        # Load asset for Issue 18/27
        tank = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tank.svg")
        # Apply fix for Issue 27: adjust tank position
        self.place_at_grid(tank, 'D6', scale_factor=0.9)
        label = Text("Water Tank", font_size=18).next_to(tank, UP)
        self.play(Create(tank), Write(label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(ORANGE)
        sum_label = Text("Integral = Sum of Slices", font_size=20, color=ORANGE)
        # Apply fix for Issue 28: adjust sum_label position
        self.place_at_grid(sum_label, 'F2', scale_factor=0.7)
        self.play(Write(sum_label))
        self.wait(2)
