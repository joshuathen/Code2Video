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
            "Measurement forces a system to choose.",
            "Born’s rule predicts the final state.",
            "Observation collapses the quantum wavefunction."
        ]
        self.setup_layout("The Measurement Problem & Collapse", lecture_lines)
        
        # Load SVG Assets
        sensor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg")
        detector = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/detector.svg")
        monitor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/monitor.svg")
        
        # Define visual elements (White wave, Cyan flash, Green vector)
        # Fixed using hex code as CYAN is not a standard Manim color constant
        wave = FunctionGraph(lambda x: 0.5 * np.sin(4*x) * np.exp(-0.5*x**2), x_range=[-3, 3], color=WHITE)
        flash = Dot(color="#00FFFF")
        vec = Vector([0, 1], color=GREEN)
        
        # Positions based on reviewer feedback
        self.place_in_area(wave, 'B4', 'D6', scale_factor=0.9)
        self.place_at_grid(sensor, 'C3', scale_factor=0.5)
        
        self.place_at_grid(flash, 'C4', scale_factor=0.6)
        self.place_at_grid(detector, 'C4', scale_factor=0.5)
        
        self.place_at_grid(vec, 'D4', scale_factor=0.8)
        self.place_at_grid(monitor, 'D4', scale_factor=0.5)
        
        # Initially hide elements
        flash.set_opacity(0)
        vec.set_opacity(0)
        detector.set_opacity(0)
        monitor.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Create(wave), FadeIn(sensor))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(FadeOut(wave), FadeOut(sensor), FadeIn(flash), FadeIn(detector))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        self.play(FadeOut(flash), FadeOut(detector), FadeIn(vec), FadeIn(monitor))
        
        self.wait(2)
