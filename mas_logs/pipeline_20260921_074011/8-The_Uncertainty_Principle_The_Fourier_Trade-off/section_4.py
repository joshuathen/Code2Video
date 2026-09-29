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
        lecture_lines = ["The Gaussian is the optimal signal.", "It acts as a perfect window.", "Engineers use it to balance domains.", "Detecting short chirps requires this precision.", "It provides the ideal processing balance."]
        self.setup_layout("The Optimal Solution: Gabor Atoms", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Fade in the Gaussian function (#FF5733) as the envelope.
        gaussian = FunctionGraph(lambda x: np.exp(-x**2), x_range=[-3, 3], color="#FF5733")
        self.place_in_area(gaussian, 'A2', 'B5', scale_factor=0.6)
        self.play(Create(gaussian))
        self.lecture[0].set_color("#FF5733")

        # === Animation for Lecture Line 2 ===
        # Modulate the Gaussian with a sine carrier wave (#33FF57).
        sine_carrier = FunctionGraph(lambda x: np.exp(-x**2) * np.sin(5 * x), x_range=[-3, 3], color="#33FF57")
        self.place_in_area(sine_carrier, 'D2', 'E5', scale_factor=0.6)
        self.play(Create(sine_carrier))
        self.lecture[1].set_color("#33FF57")

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF5733")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Overlay text and indicate optimal Gabor atom.
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        gabor_text = Text("Gabor Atom", font_size=24, color=WHITE)
        self.place_at_grid(gabor_text, 'C3')
        
        # Scale the Gaussian width to match the optimal bound.
        self.play(
            gaussian.animate.stretch(0.5, dim=0), 
            sine_carrier.animate.stretch(0.5, dim=0),
            FadeIn(gabor_text)
        )
        self.lecture[3].set_color("#33FF57")
        
        # === Animation for Lecture Line 5 ===
        # Flash the result.
        self.play(Flash(sine_carrier, color=WHITE))
        self.lecture[4].set_color("#FFFFFF")
        self.wait(2)
