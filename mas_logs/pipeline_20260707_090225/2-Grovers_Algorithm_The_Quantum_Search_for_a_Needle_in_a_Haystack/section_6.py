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

class Section6Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            'Grover’s algorithm provides a massive quadratic speedup.',
            'A million items require only one thousand quantum steps.',
            'This efficiency transforms cryptography and complex logistics optimization.'
        ]
        self.setup_layout("The Quantum Advantage", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Description: The text 'Classical: O(N)' in Red (#FF0000) appears on the left.
        self.play(self.lecture[0].animate.set_color(RED))
        
        classical_label = Text("Classical: ", font_size=32, color="#FF0000")
        classical_math = Text("O(N)", font_size=32, color="#FF0000")
        classical_group = VGroup(classical_label, classical_math).arrange(RIGHT, buff=0.1)
        
        # Fix for Issue #35: Moving classical_group to B2-B3 to avoid overlap with lecture text
        self.place_in_area(classical_group, 'B2', 'B3', scale_factor=0.9)
        self.play(Write(classical_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Description: The text 'Quantum: O(√N)' in Green (#00FF00) appears on the right.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(GREEN)
        )
        
        quantum_label = Text("Quantum: ", font_size=32, color="#00FF00")
        quantum_math = Text("O(√N)", font_size=32, color="#00FF00")
        quantum_group = VGroup(quantum_label, quantum_math).arrange(RIGHT, buff=0.1)
        
        # Fix for Issue #35: Using B4-B6 as requested
        self.place_in_area(quantum_group, 'B4', 'B6', scale_factor=0.9)
        self.play(Write(quantum_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Description: A direct comparison '1,000,000 vs 1,000' flashes to emphasize the speedup.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        comp_left = Text("1,000,000", font_size=36, color="#FF0000")
        comp_vs = Text("vs", font_size=28, color=WHITE)
        comp_right = Text("1,000", font_size=36, color="#00FF00")
        
        # Fix for Issue #33: Moving comp_left to D3
        self.place_at_grid(comp_left, 'D3', scale_factor=1.0)
        # Fix for Issue #34: Moving comp_vs and comp_right to balance the row
        self.place_in_area(comp_vs, 'D4', 'D5', scale_factor=1.0)
        self.place_at_grid(comp_right, 'D6', scale_factor=1.0)
        
        comparison_group = VGroup(comp_left, comp_vs, comp_right)
        
        # Flashing effect as requested
        for _ in range(3):
            self.play(FadeIn(comparison_group), run_time=0.3)
            self.play(FadeOut(comparison_group), run_time=0.2)
        
        self.play(FadeIn(comparison_group))
        self.wait(3)

        # Wrap up: Reset highlights
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
