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
            "Imagine two robotic arms, each with placement errors.",
            "Each arm’s error follows a bell-shaped distribution.",
            "What happens when we add these errors together?",
            "Visualize two separate normal curves side by side.",
            "They represent independent sources of randomness."
        ]
        self.setup_layout("Introduction: The Intuition of Summing Randomness", lecture_lines)
        
        # Load assets
        arm_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/arm.svg"
        robot_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg"
        
        # Mobjects
        arm1 = SVGMobject(arm_asset).set_color(WHITE)
        arm2 = SVGMobject(arm_asset).set_color(WHITE)
        robot = SVGMobject(robot_asset).set_color(WHITE)
        
        curve1 = FunctionGraph(lambda x: np.exp(-x**2), x_range=[-2, 2]).set_color("#FFCC00")
        curve2 = FunctionGraph(lambda x: np.exp(-x**2), x_range=[-2, 2]).set_color("#FFCC00")
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(arm1), FadeIn(arm2), self.lecture[0].animate.set_color(WHITE))
        self.place_at_grid(arm1, "D2", scale_factor=0.5)
        self.place_at_grid(arm2, "D5", scale_factor=0.5)
        
        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(curve1), FadeIn(curve2), self.lecture[1].animate.set_color("#FFCC00"))
        self.place_at_grid(curve1, "B2", scale_factor=0.5)
        self.place_at_grid(curve2, "B5", scale_factor=0.5)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFCC00"))
        self.play(FadeIn(robot))
        self.place_at_grid(robot, "E4", scale_factor=0.5)
        
        # === Animation for Lecture Line 4 ===
        self.play(curve1.animate.move_to(self.grid["B3"]), curve2.animate.move_to(self.grid["B4"]), self.lecture[3].animate.set_color(WHITE))
        
        # === Animation for Lecture Line 5 ===
        self.play(curve1.animate.set_color("#00FF00"), curve2.animate.set_color("#00FF00"), self.lecture[4].animate.set_color("#00FF00"))
        self.wait(2)
