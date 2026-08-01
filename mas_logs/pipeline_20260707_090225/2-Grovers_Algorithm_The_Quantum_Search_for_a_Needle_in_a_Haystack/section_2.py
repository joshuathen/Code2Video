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
        # Initialize Scene
        lecture_lines = [
            'Quantum computers represent all items as a state vector.', 
            'We create a uniform superposition of every possible state.', 
            'Initially, every item has the same probability amplitude.'
        ]
        self.setup_layout("Prerequisite: Superposition & State Vectors", lecture_lines)

        # Assets
        # Loading monitor asset as per Issue 21
        monitor = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/monitor.svg", color=WHITE)
        self.place_in_area(monitor, "C1", "F6", scale_factor=3.5)

        # Pre-create bars for efficiency
        # Using 400 bars to maintain high visual density while ensuring render speed
        num_bars = 400 
        bars = VGroup(*[
            Line(ORIGIN, UP * 0.01, stroke_width=1.5, color="#ADD8E6") 
            for _ in range(num_bars)
        ])
        bars.arrange(RIGHT, buff=0.005)
        # Position bars within the monitor area per Issue 25
        self.place_in_area(bars, "D1", "F6")
        
        # === Animation for Lecture Line 1 ===
        # Highlight current line
        self.play(self.lecture[0].animate.set_color("#FFFF00"), run_time=0.5)
        
        # A bar chart appears on a monitor [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/monitor.svg] with thin, zero-height bars.
        self.play(FadeIn(monitor), FadeIn(bars))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition highlight
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFF00"),
            run_time=0.5
        )
        
        # All bars simultaneously rise to a height representing 1/sqrt(1000)
        # Stretching bars vertically to fill the monitor screen
        self.play(
            bars.animate.stretch_to_fit_height(1.8, about_edge=DOWN),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition highlight
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00"),
            run_time=0.5
        )
        
        # A text label 'Uniform Superposition' appears in White (#FFFFFF) above the bars.
        label = Text("Uniform Superposition", font_size=28, color="#FFFFFF")
        # Apply Fix from Issue 24 and 26: center label in row B
        self.place_in_area(label, 'B1', 'B6', scale_factor=0.8)
        
        self.play(Write(label))
        self.wait(2)
