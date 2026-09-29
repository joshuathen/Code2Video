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
        self.setup_layout("Visualizing Dispersion: The Cauchy Equation Graph", [
            "Refractive index varies with wavelength.",
            "Shorter wavelengths bend more than longer ones.",
            "Prisms separate colors due to this dispersion."
        ])
        
        # Define graph axes
        axes = Axes(
            x_range=[0.4, 0.8, 0.1],
            y_range=[1.45, 1.55, 0.02],
            axis_config={"include_numbers": False},
            x_length=4,
            y_length=3
        ).set_color(WHITE)
        
        # Cauchy-like curve: n(lambda) = A + B/lambda^2
        curve = axes.plot(lambda x: 1.46 + 0.005 / (x**2), color=YELLOW)
        
        labels = axes.get_axis_labels(x_label="\\lambda", y_label="n")
        
        # Load asset
        prism_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")
        
        # Place graph and asset
        graph_group = VGroup(axes, curve, labels)
        self.place_in_area(graph_group, 'C3', 'E6', scale_factor=0.65)
        self.place_at_grid(prism_icon, 'B1', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(axes), Create(labels), FadeIn(prism_icon), run_time=1.5)
        self.play(Create(curve), run_time=2)
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        dot_blue = Dot(color=BLUE).move_to(curve.point_from_proportion(0.1))
        self.play(FadeIn(dot_blue))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        cauchy_label = Text("Cauchy Curve", font_size=20, color="#00FF00")
        self.place_at_grid(cauchy_label, 'B4', scale_factor=0.7)
        self.play(Write(cauchy_label))
        self.wait(2)
