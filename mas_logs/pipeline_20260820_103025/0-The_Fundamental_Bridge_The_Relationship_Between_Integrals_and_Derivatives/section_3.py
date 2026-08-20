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
        lecture_lines = [
            "Accumulating rate returns the original function.",
            "Volume growth is the rate of flow.",
            "The area function's rate is the curve.",
            "Fundamental Theorem links accumulation and change.",
            "Integration undoes differentiation's work."
        ]
        self.setup_layout("The Fundamental Theorem of Calculus (Part I)", lecture_lines)
        
        # Define objects
        axes = Axes(x_range=[0, 5, 1], y_range=[0, 3, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.5 * x + 0.5, color="#FFFFFF")
        curve_label = MathTex("f(t)", color="#FFFFFF").scale(0.7)
        
        area_func = Polygon(*[axes.c2p(0, 0), *[axes.c2p(x, 0.5 * x + 0.5) for x in np.linspace(0, 2, 20)], axes.c2p(2, 0)], color="#00FFFF", fill_opacity=0.4)
        area_label = MathTex("A(x)", color="#00FFFF").scale(0.7)
        
        # Assets
        speedometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.place_at_grid(axes, 'B4', scale_factor=0.5)
        self.place_at_grid(curve, 'B4', scale_factor=0.5)
        self.place_at_grid(curve_label, 'A4', scale_factor=0.8) # Fixed per issue 27
        self.place_at_grid(speedometer, 'B5', scale_factor=0.7)
        self.play(Create(axes), Create(curve), Write(curve_label), FadeIn(speedometer))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        self.place_in_area(area_func, 'B2', 'C5', scale_factor=0.8) # Fixed per issue 26
        self.place_at_grid(area_label, 'C4', scale_factor=0.7)
        self.play(FadeIn(area_func), Write(area_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFCC00")
        derivative_label = MathTex("A'(x) = f(x)", color="#FFCC00").scale(0.7)
        self.place_at_grid(derivative_label, 'D3', scale_factor=1.0)
        self.play(Write(derivative_label))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#AA00FF")
        ftc_text = Text("FTC", color="#AA00FF", font_size=24)
        self.place_at_grid(ftc_text, 'E4', scale_factor=1.0) # Fixed per issue 28
        self.play(FadeIn(ftc_text))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFFFF")
        self.place_at_grid(ruler, 'B3', scale_factor=0.5) # Using asset
        self.play(FadeIn(ruler))
        self.play(ruler.animate.shift(RIGHT * 2))
        self.wait(1)
