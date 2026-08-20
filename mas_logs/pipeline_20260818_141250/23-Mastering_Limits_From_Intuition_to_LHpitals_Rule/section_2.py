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
        self.setup_layout("The Challenge of Indeterminate Forms", [
            "Direct substitution often fails at indeterminate forms.",
            "The drone function hits a gap.",
            "We inspect the slope near the gap."
        ])
        
        # Load asset
        drone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg", color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        # Display indeterminate fraction 0/0 #FFFFFF
        fraction = MathTex(r"\\frac{0}{0}", color=WHITE)
        self.place_in_area(fraction, 'B2', 'C3', scale_factor=1.5)
        self.place_at_grid(drone.copy().scale(0.5), 'B4')
        self.play(Write(fraction), FadeIn(drone.copy()))
        self.lecture[0].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show divergent paths #FFD700 approaching the point
        path1 = Line(start=self.grid['D1'], end=self.grid['E3'], color=GOLD)
        path2 = Line(start=self.grid['D5'], end=self.grid['E3'], color=GOLD)
        dot = Dot(self.grid['E3'], color=RED)
        
        # drone_animation area
        drone_group = VGroup(path1, path2, dot)
        self.place_in_area(drone_group, 'D2', 'F4', scale_factor=1.0)
        
        self.play(Create(path1), Create(path2), FadeIn(dot))
        self.lecture[1].set_color(GOLD)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate ratio #00FFFF crashing or oscillating
        ratio_text = Text("Ratio", color="#00FFFF", font_size=24)
        self.place_at_grid(ratio_text, 'E4', scale_factor=1.0)
        
        drone_icon = drone.copy()
        self.place_at_grid(drone_icon, 'E3', scale_factor=0.3)
        
        # Animate drone with ratio label
        self.play(
            Indicate(dot, color="#00FFFF"),
            Write(ratio_text),
            drone_icon.animate.shift(UP * 0.5),
            run_time=2
        )
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
