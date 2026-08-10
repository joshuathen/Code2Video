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
        self.setup_layout("Boundary and Initial Conditions", [
            "PDEs need constraints to solve.",
            "Initial conditions define the starting state.",
            "Boundary conditions lock the edges."
        ])

        # Assets
        rod = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg")
        self.place_in_area(rod, 'C1', 'E6', scale_factor=0.8)
        
        # We need specific points to color based on the storyboard
        # The svg might be complex, so we add dots on top for clear boundary markers
        boundary_left = Dot(color=WHITE)
        boundary_right = Dot(color=WHITE)
        
        # Position markers manually relative to the rod's area (roughly)
        # Using the area grid coordinates for alignment
        self.place_at_grid(boundary_left, 'C2')
        self.place_at_grid(boundary_right, 'C6')

        # Initial state pulse
        pulse = FunctionGraph(lambda x: 0.3 * np.exp(-5 * (x)**2), x_range=[-1.0, 1.0], color=WHITE)
        self.place_in_area(pulse, 'C2', 'C6', scale_factor=1.0)
        pulse.shift(UP * 0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00CED1"))
        self.play(FadeIn(rod), FadeIn(boundary_left), FadeIn(boundary_right))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        self.play(Create(pulse))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF4500"))
        self.play(boundary_left.animate.set_color("#FF4500"), boundary_right.animate.set_color("#FF4500"))
        self.wait(1)
