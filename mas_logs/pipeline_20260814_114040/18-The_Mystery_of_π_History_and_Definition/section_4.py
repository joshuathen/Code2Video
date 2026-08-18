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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Practical Application: Why π Matters", [
            "Pi is essential for modern technology.",
            "It powers calculations for rotation and waves.",
            "Robots use pi for precise movement.",
            "GPS satellites rely on pi's accuracy.",
            "Our world runs on pi's precision."
        ])
        
        # Load Assets
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        satellite = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/satellite.svg")
        earth = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/earth.svg")
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(robot, "B4", scale_factor=0.6)
        self.play(FadeIn(robot), self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Rotate(robot, angle=PI/4, about_point=robot.get_center()))

        # === Animation for Lecture Line 2 ===
        wave = FunctionGraph(lambda x: 0.5 * np.sin(3 * x), x_range=[-1, 1], color="#FF00FF")
        self.place_at_grid(wave, "C3", scale_factor=0.7)
        self.play(Create(wave), self.lecture[1].animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.play(robot.animate.shift(RIGHT*0.5))

        # === Animation for Lecture Line 4 ===
        self.place_at_grid(satellite, "E3", scale_factor=0.4)
        self.place_at_grid(earth, "E5", scale_factor=0.4)
        path = Line(satellite.get_center(), earth.get_center(), color="#00FFFF")
        self.play(FadeIn(satellite), FadeIn(earth), Create(path), self.lecture[3].animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 5 ===
        pi_text = MathTex(r"\\pi", color="#FFFF00", font_size=72)
        self.place_at_grid(pi_text, "C5", scale_factor=1.0)
        self.play(Write(pi_text), self.lecture[4].animate.set_color("#FFFF00"))
        
        # Scaling systems
        self.play(
            FadeOut(robot), FadeOut(wave), FadeOut(satellite), FadeOut(earth), FadeOut(path),
            pi_text.animate.scale(2)
        )
