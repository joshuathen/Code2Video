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
        lecture_lines = ["Superposition builds complex periodic shapes.", "Sum multiple sine waves of varying frequencies.", "Small rotations combine into larger patterns.", "Epicycles stack to trace complex silhouettes.", "Any curve emerges from simple circles."]
        self.setup_layout("Fourier's Core Concept: Building Complexity", lecture_lines)
        
        # Load SVG assets
        pendulum_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pendulum.svg")
        motor_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/motor.svg")

        # Elements
        wave_a = FunctionGraph(lambda x: 0.5 * np.sin(x), x_range=[-2, 2], color="#3498DB")
        label_a = Text("A", font_size=20, color="#3498DB")
        
        wave_b = FunctionGraph(lambda x: 0.5 * np.sin(2 * x), x_range=[-2, 2], color="#E74C3C")
        label_b = Text("B", font_size=20, color="#E74C3C")
        
        wave_c = FunctionGraph(lambda x: 0.3 * np.sin(3 * x), x_range=[-2, 2], color="#2ECC71")
        label_c = Text("C", font_size=20, color="#2ECC71")
        
        wave_sum = FunctionGraph(lambda x: 0.5 * np.sin(x) + 0.5 * np.sin(2 * x) + 0.3 * np.sin(3 * x), x_range=[-2, 2], color="#F1C40F")
        label_sum = Text("Sum", font_size=20, color="#F1C40F")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#3498DB"))
        self.place_at_grid(wave_a, 'C2')
        self.place_at_grid(label_a, 'B2', scale_factor=0.8)
        self.place_at_grid(pendulum_icon, 'A2', scale_factor=0.3)
        self.play(Create(wave_a), Write(label_a), FadeIn(pendulum_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#E74C3C"))
        self.place_at_grid(wave_b, 'C4')
        self.place_at_grid(label_b, 'B4', scale_factor=0.8)
        self.play(Create(wave_b), Write(label_b))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#9B59B6"))
        wave_combined = FunctionGraph(lambda x: 0.5 * np.sin(x) + 0.5 * np.sin(2 * x), x_range=[-2, 2], color="#9B59B6")
        self.place_in_area(wave_combined, 'D3', 'E4', scale_factor=0.9)
        self.play(FadeIn(wave_combined))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#2ECC71"))
        self.place_at_grid(wave_c, 'C6')
        self.place_at_grid(label_c, 'B6', scale_factor=0.8)
        self.play(Create(wave_c), Write(label_c))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#F1C40F"))
        self.place_in_area(wave_sum, 'D5', 'E6', scale_factor=0.9)
        self.place_at_grid(label_sum, 'C5', scale_factor=0.8)
        self.place_at_grid(motor_icon, 'F5', scale_factor=0.3)
        self.play(Create(wave_sum), Write(label_sum), FadeIn(motor_icon))
        
        self.wait(2)
