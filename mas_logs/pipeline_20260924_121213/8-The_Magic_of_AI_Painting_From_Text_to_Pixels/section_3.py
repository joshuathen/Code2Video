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
        self.setup_layout("The Mathematics: Probability & Gradients", [
            "Models predict noise residual using U-Net.",
            "We minimize the difference between predicted noise.",
            "Guidance scales balance text and image fidelity.",
            "Move slider to change strictness of prompt.",
            "This defines imagination versus image structure."
        ])
        
        # === Animation for Lecture Line 1 ===
        axes = Axes(x_range=[-3, 3], y_range=[0, 1], axis_config={"include_numbers": False}).scale(0.5)
        curve = axes.plot(lambda x: np.exp(-x**2), color="#00FF00")
        self.place_at_grid(VGroup(axes, curve), 'B5', scale_factor=0.6)
        self.play(Create(curve), run_time=1.5)
        self.lecture[0].set_color("#00FF00")

        # === Animation for Lecture Line 2 ===
        vec = Arrow(start=ORIGIN, end=RIGHT*0.5 + UP*0.5, color="#FFFF00")
        self.place_at_grid(vec, 'C5', scale_factor=0.6)
        self.play(GrowArrow(vec), run_time=1)
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/slider.svg]
        slider = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/slider.svg", color="#FF00FF")
        self.place_at_grid(slider, 'D4', scale_factor=0.7)
        self.play(FadeIn(slider), run_time=1)
        self.lecture[2].set_color("#FF00FF")

        # === Animation for Lecture Line 4 ===
        # Move loss_curve to F4 as requested (Critic #29)
        loss_curve = Line(LEFT, RIGHT, color=WHITE).scale(0.5)
        self.place_at_grid(loss_curve, 'F4', scale_factor=0.9)
        self.play(Create(loss_curve), run_time=1)
        self.lecture[3].set_color(WHITE)

        # === Animation for Lecture Line 5 ===
        # Math objective: move to E5 as requested (Critic #28)
        eq = MathTex(r"L = ||\epsilon - \epsilon_\theta||^2", color=WHITE)
        self.place_at_grid(eq, 'E5', scale_factor=0.8)
        self.play(Write(eq), run_time=1.5)
        self.lecture[4].set_color(WHITE)
        self.wait(2)
