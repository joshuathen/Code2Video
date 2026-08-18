from manim import *
import os

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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary & Application", [
            "Summing independent gaussians is a closed operation.",
            "This property is crucial for sensor fusion.",
            "It predicts combined error in real-world systems."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Table showing summation rule: N1 + N2 = N_sum
        table = VGroup(
            Text("N(μ1, σ1²)", font_size=24, color=BLUE),
            Text("+", font_size=24, color=WHITE),
            Text("N(μ2, σ2²)", font_size=24, color=BLUE),
            Text("=", font_size=24, color=WHITE),
            Text("N(μ1+μ2, σ1²+σ2²)", font_size=24, color=GREEN)
        ).arrange(RIGHT, buff=0.2)
        self.place_at_grid(table, 'C2', scale_factor=0.8)
        self.play(FadeIn(table))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Visualization of Gaussian curves combining
        axes = Axes(x_range=[-3, 3, 1], y_range=[0, 1, 0.5], axis_config={"include_tip": False}).scale(0.4)
        curve1 = axes.plot(lambda x: np.exp(-x**2/0.5), color=BLUE)
        curve2 = axes.plot(lambda x: np.exp(-(x-0.5)**2/0.5), color=BLUE)
        combined_curve = axes.plot(lambda x: np.exp(-x**2/0.8), color="#90EE90")
        
        plot_group = VGroup(axes, curve1, curve2, combined_curve)
        self.place_in_area(plot_group, 'E4', 'F6', scale_factor=0.7)
        
        self.play(Create(axes), Create(curve1), Create(curve2))
        self.play(ReplacementTransform(VGroup(curve1, curve2), combined_curve))
        self.lecture[1].set_color("#90EE90")

        # === Animation for Lecture Line 3 ===
        # Text/icon for Error Margin
        error_label = Text("Error = Combined Noise", font_size=24, color=YELLOW)
        self.place_at_grid(error_label, 'D4', scale_factor=0.6)
        
        # Load asset
        sensor_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg")
        self.place_at_grid(sensor_icon, 'E2', scale_factor=0.4)
        
        self.play(Write(error_label), FadeIn(sensor_icon))
        self.lecture[2].set_color(YELLOW)
        
        self.wait(2)
        self.play(FadeOut(table), FadeOut(plot_group), FadeOut(error_label), FadeOut(sensor_icon))
