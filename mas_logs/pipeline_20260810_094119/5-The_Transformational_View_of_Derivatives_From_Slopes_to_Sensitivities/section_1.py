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
        self.setup_layout("Introduction: The Frozen Snapshot vs. The Flow", [
            "Derivatives move beyond static slopes to dynamic growth.",
            "[Asset: Cheetah_Path] represents the evolving function over time.",
            "[Asset: Derivative_Operator] acts like a speedometer for change.",
            "[Asset: Velocity_Vector] captures the instantaneous growth rate.",
            "This reveals the function's DNA of motion."
        ])
        
        # Animations
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        curve = FunctionGraph(lambda x: 0.1 * x**2, x_range=[-3, 3], color=WHITE)
        self.place_in_area(curve, 'B2', 'C5', scale_factor=0.6)
        
        cheetah = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        self.place_at_grid(cheetah, 'A6', scale_factor=0.5)
        self.play(Create(curve), FadeIn(cheetah))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        dot = Dot(color="#FF0000")
        self.place_at_grid(dot, 'B3', scale_factor=0.5)
        self.play(FadeIn(dot))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        tangent = Line(start=[-1, -0.2, 0], end=[1, 0.2, 0], color="#FFFF00").scale(0.8)
        tangent.move_to(dot.get_center())
        self.play(Create(tangent))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        vector = Arrow(start=dot.get_center(), end=dot.get_center() + RIGHT*0.8, color="#00FF00", buff=0)
        self.play(GrowArrow(vector))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FFFF")
        speedometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg")
        self.place_at_grid(speedometer, 'D4', scale_factor=0.7)
        self.play(FadeOut(tangent), FadeOut(vector), FadeIn(speedometer))
