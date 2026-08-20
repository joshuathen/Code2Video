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
            "The Wave Equation models oscillations on a string.",
            "It captures how waves move through space and time.",
            "Visualize a sine-wave string vibrating continuously."
        ]
        self.setup_layout("The Wave Equation", lecture_lines)
        
        # Load asset
        string_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/string.svg")
        
        # Setup Sine Wave
        wave = FunctionGraph(lambda x: 0.5 * np.sin(2 * np.pi * x), x_range=[-1, 1], color="#00FFFF")
        self.place_in_area(wave, 'C2', 'F5', scale_factor=0.75)
        
        amplitude_label = Text("Amplitude", font_size=20, color="#FFFFFF")
        self.place_at_grid(amplitude_label, 'C1', scale_factor=0.7)
        
        # Trackers for animation
        time_tracker = ValueTracker(0)
        
        def update_wave(m):
            t = time_tracker.get_value()
            m.become(FunctionGraph(lambda x: 0.5 * np.sin(2 * np.pi * x + t), x_range=[-1, 1], color="#00FFFF"))
            self.place_in_area(m, 'C2', 'F5', scale_factor=0.6)

        wave.add_updater(update_wave)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.play(FadeIn(string_icon), Create(wave))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        self.play(Write(amplitude_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        self.play(time_tracker.animate.set_value(2 * np.pi), run_time=3, rate_func=linear)
        
        self.wait(1)
