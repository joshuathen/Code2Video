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
        lecture_lines = [
            "The second derivative f''(x) is the derivative of f'(x).",
            "It measures acceleration, or how velocity changes.",
            "Graphically, it determines the curve's concavity."
        ]
        self.setup_layout("Defining the Second Derivative", lecture_lines)
        
        # Load assets
        tachometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tachometer.svg")
        pedal = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pedal.svg")
        car = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        
        # Setup Axes for visual
        axes = Axes(x_range=[-2, 2], y_range=[-1, 3], x_length=3, y_length=2).shift(self.grid["C4"])
        curve = axes.plot(lambda x: x**2, color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        # The second derivative f''(x) is the derivative of f'(x).
        self.lecture[0].set_color("#00FFFF")
        self.place_at_grid(tachometer, "B5", scale_factor=0.25)
        tachometer.set_color("#00FFFF")
        label1 = MathTex("f''(x)", color="#00FFFF").next_to(tachometer, UP)
        self.add(tachometer, label1)
        self.play(FadeIn(tachometer), Write(label1))
        
        # === Animation for Lecture Line 2 ===
        # It measures acceleration, or how velocity changes.
        self.lecture[1].set_color("#FF9900")
        self.place_at_grid(pedal, "D5", scale_factor=0.25)
        pedal.set_color("#FF9900")
        label2 = Text("Acceleration", font_size=16, color="#FF9900").next_to(pedal, DOWN)
        self.add(pedal, label2)
        self.play(FadeIn(pedal), Write(label2))
        
        # === Animation for Lecture Line 3 ===
        # Graphically, it determines the curve's concavity.
        self.lecture[2].set_color("#FF0000")
        self.add(axes, curve)
        self.place_at_grid(car, "F6", scale_factor=0.2)
        car.set_color("#FF0000")
        
        self.play(
            MoveAlongPath(car, curve, rate_func=linear),
            run_time=3
        )
        self.wait(1)
