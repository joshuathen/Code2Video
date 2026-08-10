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
        self.setup_layout("The Heat Equation: A Core Example", [
            "The heat equation models diffusion.",
            "Heat flows from hot to cold regions.",
            "A sharp peak smooths over time.",
            "It is a hill leveling out.",
            "PDEs predict this spreading energy."
        ])

        # Assets
        thermometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/thermometer.svg")
        radiator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/radiator.svg")
        
        # Setup Math
        heat_eq = MathTex(r"u_t", "=", r"\alpha", r"u_{xx}", color=WHITE)
        self.place_at_grid(heat_eq, 'C3', scale_factor=1.0)

        # Setup Plot
        axes = Axes(x_range=[-3, 3, 1], y_range=[0, 3, 1], axis_config={"include_tip": False}, x_length=4, y_length=2.5)
        self.place_in_area(axes, 'D2', 'F5', scale_factor=0.9)
        
        # Initial curve
        def get_curve(t):
            return axes.plot(lambda x: 2 * np.exp(-(x**2) / (0.1 + 0.5 * t)), color="#FF4500")
        
        curve = get_curve(0)
        
        # Group assets and visuals
        plot_group = VGroup(axes, curve)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF4500"))
        self.place_at_grid(thermometer, 'B6', scale_factor=0.5)
        self.play(FadeIn(thermometer), FadeIn(curve))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        self.play(FadeIn(heat_eq))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.place_at_grid(radiator, 'D6', scale_factor=0.5)
        self.play(FadeIn(radiator))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FFFF"))
        self.play(Transform(curve, get_curve(2)), run_time=3)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF00FF"))
        self.play(Indicate(heat_eq))
        self.wait(1)
