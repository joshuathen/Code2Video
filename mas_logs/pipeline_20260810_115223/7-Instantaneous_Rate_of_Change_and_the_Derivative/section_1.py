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
        self.setup_layout("Prerequisite: The Average Rate of Change", [
            "Average speed is total distance over total time.",
            "Secant lines show average slope between two points.",
            "But cheetahs change speed throughout the sprint."
        ])
        
        # Setup curve and points
        axes = Axes(x_range=[0, 5], y_range=[0, 10], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.4 * x**2, color=WHITE)
        p1 = axes.c2p(1, 0.4)
        p2 = axes.c2p(4, 6.4)
        dot1 = Dot(p1, color=RED)
        dot2 = Dot(p2, color=BLUE)
        secant = Line(p1, p2, color="#FFD700")
        
        cheetah = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        
        group = VGroup(axes, curve, dot1, dot2, secant, cheetah)
        # Using A2 to E5 to balance the space, incorporating the cheetah asset.
        self.place_in_area(group, 'A2', 'E5', scale_factor=0.6)
        
        # Cheetah placement relative to points
        cheetah.next_to(dot2, UR, buff=0.2).scale(1.5)
        
        # Labels
        label = Text("Average Rate of Change", color="#87CEEB", font_size=24)
        self.place_at_grid(label, 'E6', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(group), FadeIn(label))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#87CEEB"))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(RED))
        self.wait(1)
