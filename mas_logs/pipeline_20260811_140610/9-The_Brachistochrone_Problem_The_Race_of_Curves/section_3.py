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
        self.setup_layout("The Solution: Enter the Cycloid", [
            "The solution is the unique cycloid curve.",
            "Think of a point on a rolling circle.",
            "The cycloid path optimizes energy conversion.",
            "It steals time to reach high speeds early.",
            "Nature prefers this elegant, efficient trajectory."
        ])
        
        # Define the cycloid path
        # x = r(t - sin(t)), y = r(1 - cos(t))
        r = 0.5
        cycloid_func = lambda t: np.array([r * (t - np.sin(t)), -r * (1 - np.cos(t)), 0])
        cycloid_curve = ParametricFunction(cycloid_func, t_range=[0, 2*PI], color="#FF33FF")
        
        # Assets
        circle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg")
        particle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF33FF")
        # Applying requested layout fix from issue 27/42/29/44
        self.place_in_area(cycloid_curve, "A4", "C6", scale_factor=0.5)
        self.play(Create(cycloid_curve))
        self.place_at_grid(circle_icon, "A5", scale_factor=0.6)
        self.play(FadeIn(circle_icon))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        eq = MathTex(r"x=r(\theta-\sin\theta), y=r(1-\cos\theta)", font_size=24)
        # Applying requested layout fix from issue 28/43
        self.place_at_grid(eq, "E4", scale_factor=0.7)
        self.play(Write(eq))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF5733")
        value_tracker = ValueTracker(0)
        particle = particle_icon.copy().scale(0.3).set_color("#FF5733")
        particle.add_updater(lambda d: d.move_to(cycloid_curve.point_from_proportion(value_tracker.get_value())))
        self.add(particle)
        self.play(value_tracker.animate.set_value(1), run_time=3, rate_func=linear)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        line = Line(start=self.grid["B4"], end=self.grid["C6"], color=WHITE)
        self.play(Create(line))
        self.play(value_tracker.animate.set_value(0), run_time=2)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FFFF")
        self.play(Indicate(cycloid_curve, color="#00FFFF"))
        self.play(particle.animate.set_color("#00FFFF"))
