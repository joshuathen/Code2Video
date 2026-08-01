from manim import *
import numpy as np
from pathlib import Path

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        # Focused fix for FileExistsError: manually ensure the text directory exists with exist_ok=True
        Path(config.get_dir("text_dir")).mkdir(parents=True, exist_ok=True)
        
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        # Initialize with GRAY to allow highlighting animation
        lecture_texts = [Text(line, font_size=22, color=GRAY) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (6x6 grid on right side)
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
        # Setup basic layout
        self.setup_layout(
            "The Math of Probability (Amplitude)", 
            [
                'We use probability amplitudes to calculate these weights.', 
                'Squaring these amplitudes gives the total probability.', 
                'All probabilities must always sum up to one.'
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Display the equation |ψ⟩ = α|0⟩ + β|1⟩ centered in row B
        # Replaced MathTex with a VGroup of Text objects using unicode to avoid the 'latex' FileNotFoundError
        eq1 = VGroup(
            Text("|ψ⟩ =", font_size=32),
            Text("α", font_size=32),
            Text("|0⟩ +", font_size=32),
            Text("β", font_size=32),
            Text("|1⟩", font_size=32)
        ).arrange(RIGHT, buff=0.15)
        
        eq1[1].set_color("#FF00FF") # α in Magenta
        eq1[3].set_color("#00FFFF") # β in Cyan
        
        self.place_in_area(eq1, "B2", "B5", scale_factor=1.2)
        self.play(Write(eq1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Move focus to second lecture line
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(WHITE)
        )
        
        # Display the square of amplitudes eq2: P(0)=|α|², P(1)=|β|²
        # Replaced MathTex with a VGroup of Text objects using unicode
        eq2 = VGroup(
            Text("P(0) = |α|²", font_size=28),
            Text(",   ", font_size=28),
            Text("P(1) = |β|²", font_size=28)
        ).arrange(RIGHT, buff=0.1)
        
        eq2[0].set_color("#FF00FF")
        eq2[2].set_color("#00FFFF")
        
        self.place_in_area(eq2, "D2", "D5", scale_factor=1.1)
        self.play(FadeIn(eq2, shift=UP))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Move focus to third lecture line
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(WHITE)
        )
        
        # Display normalization eq3: |α|² + |β|² = 1
        # Replaced MathTex with Text using unicode
        eq3 = Text("|α|² + |β|² = 1", font_size=36, color=YELLOW)
        
        self.place_in_area(eq3, "F2", "F5", scale_factor=1.3)
        self.play(Write(eq3))
        self.wait(2)