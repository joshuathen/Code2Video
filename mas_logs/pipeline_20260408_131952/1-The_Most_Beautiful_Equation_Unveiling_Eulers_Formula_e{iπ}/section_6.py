from manim import *
import numpy as np
from pathlib import Path

# Fix for KeyError: 'iπ' caused by curly braces in the input file path during Manim's directory resolution.
if config.input_file:
    config.input_file = Path(str(config.input_file).replace("{", "").replace("}", ""))

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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup layout with specific title and lecture lines
        lecture_lines = [
            'From simple rotations to the heart of physics.',
            'This formula powers quantum mechanics and wireless signals.',
            "One simple equation reveals the universe's hidden connections."
        ]
        self.setup_layout("Conclusion and Real-World Echoes", lecture_lines)
        
        # Colors for lecture highlighting
        colors = ["#FFFF00", "#00FFFF", "#00FF00"]

        # === Animation for Lecture Line 1 ===
        # Equation e^{i\pi} + 1 = 0 glows in the center in #FFFFFF.
        self.play(self.lecture[0].animate.set_color(colors[0]))
        
        # Replacing MathTex with Text to avoid FileNotFoundError for 'latex'
        # We use unicode characters for the power and pi symbol.
        equation = Text("eⁱπ + 1 = 0", color=WHITE)
        self.place_in_area(equation, "C2", "D5", scale_factor=1.5)
        
        # Add a glow effect using a duplicate with slightly larger stroke
        glow = equation.copy().set_stroke(WHITE, 8).set_opacity(0.3)
        
        self.play(
            FadeIn(equation),
            FadeIn(glow),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Labels 'Quantum Mechanics' and 'Electrical Engineering' fade in at the sides.
        self.play(self.lecture[1].animate.set_color(colors[1]))
        
        label_qm = Text("Quantum Mechanics", font_size=20, color="#A020F0") # Purple
        label_ee = Text("Electrical Engineering", font_size=20, color="#FF8C00") # Dark Orange
        
        # Fixed positioning as per Issue 33 and 34
        self.place_in_area(label_qm, 'B2', 'B4', scale_factor=0.8)
        self.place_in_area(label_ee, 'E4', 'E6', scale_factor=0.8)
        
        self.play(
            FadeIn(label_qm),
            FadeIn(label_ee),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The equation and labels pulse with light before fading to black.
        self.play(self.lecture[2].animate.set_color(colors[2]))
        
        # Pulse animation
        pulse_group = VGroup(equation, glow, label_qm, label_ee)
        
        self.play(
            pulse_group.animate.scale(1.1),
            glow.animate.set_stroke(width=12, opacity=0.5),
            rate_func=there_and_back,
            run_time=1.5
        )
        
        self.wait(2)
        
        # Final fade out
        self.play(
            FadeOut(pulse_group),
            FadeOut(self.lecture),
            FadeOut(self.title),
            run_time=2
        )
        self.wait(1)
