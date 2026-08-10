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
            "Turbulence is a complex energy cascade.",
            "Large swirls break into smaller ones.",
            "Reynolds number defines inertial versus viscous forces."
        ]
        self.setup_layout("Introduction: The Chaos in the Coffee Cup", lecture_lines)
        
        # Assets
        coffee_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coffee.svg")
        self.place_at_grid(coffee_icon, 'A6', scale_factor=0.3)
        
        cup_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cup.svg")

        # Define objects
        fluid_base = VGroup(
            *[Circle(radius=0.5, color="#3498DB").shift(np.random.rand(3)) for _ in range(10)]
        )
        eddy = Circle(radius=0.8, color="#E74C3C")
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(coffee_icon), FadeIn(fluid_base), run_time=1)
        self.lecture[0].set_color("#3498DB")

        # === Animation for Lecture Line 2 ===
        # Using cup_asset as the container for the eddy as per requirement
        self.place_in_area(cup_asset, 'B3', 'E5', scale_factor=0.5)
        self.place_at_grid(eddy, 'B5', scale_factor=0.6)
        self.play(FadeIn(cup_asset), FadeIn(eddy))
        self.play(Indicate(eddy))
        self.lecture[1].set_color("#E74C3C")

        # === Animation for Lecture Line 3 ===
        re_text = MathTex(r"Re = \frac{\text{Inertial}}{\text{Viscous}}", color=YELLOW)
        self.place_in_area(re_text, 'D4', 'F6', scale_factor=0.9)
        self.play(Write(re_text))
        self.lecture[2].set_color(YELLOW)
        
        self.wait(2)
