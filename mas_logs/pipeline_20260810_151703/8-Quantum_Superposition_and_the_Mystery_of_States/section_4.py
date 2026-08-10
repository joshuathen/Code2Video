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
        self.setup_layout("Application: The Power of Parallelism", [
            "Superposition grants massive parallel processing potential.",
            "Quantum computers evaluate many paths simultaneously.",
            "This creates a powerful quantum advantage."
        ])
        
        # Assets
        processor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/processor.svg")
        computer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        
        # === Animation for Lecture Line 1 ===
        # Visualize simultaneous calculation across multiple states using processor.svg
        proc1 = processor.copy()
        self.place_in_area(proc1, "A1", "C3", scale_factor=0.6)
        
        self.lecture[0].set_color(BLUE)
        self.play(FadeIn(proc1), run_time=1.5)

        # === Animation for Lecture Line 2 ===
        # Compare classical bit operation (computer) with qubit state operation (processor)
        comp = computer.copy()
        proc2 = processor.copy()
        
        self.place_in_area(comp, "D1", "F3", scale_factor=0.5)
        self.place_in_area(proc2, "D4", "F6", scale_factor=0.5)
        
        classical_label = Text("Classical", font_size=16, color=YELLOW).next_to(comp, UP)
        quantum_label = Text("Quantum", font_size=16, color=GREEN).next_to(proc2, UP)
        
        self.lecture[1].set_color(GREEN)
        self.play(FadeIn(comp), Write(classical_label), FadeIn(proc2), Write(quantum_label), run_time=1)

        # === Animation for Lecture Line 3 ===
        # Illustrate exponential speedup using processor.svg
        proc3 = processor.copy()
        self.place_in_area(proc3, "B4", "C6", scale_factor=0.8)
        
        speedup_text = Text("2^n States", font_size=24, color=RED).next_to(proc3, DOWN)
        
        self.lecture[2].set_color(RED)
        self.play(FadeIn(proc3), Write(speedup_text), run_time=1)
        self.wait(1)
