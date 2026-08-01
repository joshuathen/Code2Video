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
        self.setup_layout("Prerequisite: The Quantum Superposition", [
            "Quantum computers use superposition to represent all states.",
            "Applying Hadamard gates creates an equal probability distribution.",
            "Every possible answer now exists in parallel."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show 10 vertical bars of equal height in #00BFFF with 'Probability Amplitude' label.
        self.play(self.lecture[0].animate.set_color("#00BFFF"))
        
        bars = VGroup(*[
            Rectangle(height=2.0, width=0.3, fill_opacity=0.8, fill_color="#00BFFF", color="#00BFFF") 
            for _ in range(10)
        ]).arrange(RIGHT, buff=0.1)
        
        # Resolved Issue 24: Shift bars away from lecture text by starting at B2
        self.place_in_area(bars, "B2", "E6")
        
        label = Text("Probability Amplitude", font_size=24, color="#00BFFF")
        # Resolved Issue 25: Center label across area A2 to A6
        self.place_in_area(label, "A2", "A6", scale_factor=0.8)
        
        self.play(Create(bars), Write(label), run_time=2)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Pulse all bars simultaneously in #FFFFFF to represent a Hadamard transform.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFFFF")
        )
        
        self.play(
            AnimationGroup(*[Indicate(bar, color="#FFFFFF", scale_factor=1.1) for bar in bars]),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Maintain the uniform heights to show the system is in a state of 'all chests partially open'.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00BFFF")
        )
        
        # Visualizing "all chests partially open" by adding a slight glow or simple highlight
        glow_bars = bars.copy().set_color("#FFFFFF").set_opacity(0.3)
        self.play(FadeIn(glow_bars), run_time=1)
        self.play(FadeOut(glow_bars), run_time=1)
        
        self.wait(3)
