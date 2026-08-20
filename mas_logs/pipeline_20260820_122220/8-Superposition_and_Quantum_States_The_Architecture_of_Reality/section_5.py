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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Superposition enables massive parallel quantum computing.",
            "Quantum algorithms explore all paths simultaneously.",
            "Possibilities are defined by complex probability amplitudes."
        ]
        self.setup_layout("Synthesis and Application", lecture_lines)
        
        # Define objects
        # Using SVG asset for qubit as requested in storyboard
        qubit = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        waves = VGroup(*[FunctionGraph(lambda x: 0.1 * np.sin(5 * x), color=c) for c in [BLUE, "#00FFFF", TEAL]])
        maze_paths = VGroup(*[Line(start=LEFT*0.5, end=RIGHT*0.5, color=GREY) for _ in range(5)])
        algo_text = Text("Quantum Computing", font_size=36, color=WHITE)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_at_grid(qubit, 'F2', scale_factor=0.6)
        self.place_in_area(waves, 'A5', 'B6', scale_factor=0.4)
        self.play(FadeIn(qubit), Create(waves))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.place_in_area(maze_paths, 'C3', 'D4', scale_factor=0.5)
        self.play(Create(maze_paths))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(TEAL))
        self.place_at_grid(algo_text, 'E5', scale_factor=0.5)
        # Re-using asset for final screen
        qubit_final = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        self.place_at_grid(qubit_final, 'F5', scale_factor=0.4)
        self.play(Write(algo_text), FadeIn(qubit_final))
        self.wait(2)
