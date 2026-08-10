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
        self.setup_layout("The Core Revelation: The Central Limit Theorem", [
            "The Central Limit Theorem defines this.",
            "Sample means approach a Normal Distribution.",
            "Valid for any original population shape.",
            "Required sample size typically n equals 30.",
            "Chaos transforms into stable bell curves."
        ])
        
        # Define objects
        # Fixed: Simplified LaTeX string to avoid compilation issues by reducing double-backslash nesting
        formula = MathTex(r"\bar{X} \sim N(\mu, \frac{\sigma^2}{n})", color=WHITE)
        normal_curve = FunctionGraph(lambda x: np.exp(-x**2/2), x_range=[-3, 3], color="#32CD32")
        cloud = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cloud.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        sample_size_label = Text("n = 5", font_size=24)
        self.place_at_grid(sample_size_label, 'C1', scale_factor=0.9)
        self.play(Write(sample_size_label))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#32CD32")
        self.place_in_area(normal_curve, 'C2', 'E4', scale_factor=0.7)
        self.place_in_area(cloud, 'C2', 'E4', scale_factor=0.7)
        self.play(FadeIn(cloud), run_time=1)
        self.play(Transform(cloud, normal_curve))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00BFFF")
        sample_size_label_2 = Text("n = 10", font_size=24)
        self.play(Transform(sample_size_label, sample_size_label_2))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF4500")
        sample_size_label_3 = Text("n = 30", font_size=24)
        self.play(Transform(sample_size_label, sample_size_label_3))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF00FF")
        self.place_at_grid(formula, 'B4', scale_factor=0.9)
        self.play(FadeIn(formula))
        self.play(Flash(formula, color="#FFFFFF", flash_radius=0.5))
        self.wait(1)
