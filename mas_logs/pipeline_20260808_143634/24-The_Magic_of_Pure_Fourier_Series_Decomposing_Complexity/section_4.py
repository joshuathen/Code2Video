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
        self.setup_layout("Visualizing the Synthesis", [
            "Add harmonics to reconstruct a target signal.",
            "Summing sine waves approximates complex shapes.",
            "More harmonics lead to a sharper approximation."
        ])
        
        # Colors
        color_1 = "#00FFFF"
        color_2 = "#FF0000"
        color_3 = "#00FF00"
        
        # Setup Plot area (right side)
        axes = Axes(x_range=[-PI, PI], y_range=[-2, 2], x_length=5, y_length=4)
        self.place_in_area(axes, "B3", "E5", scale_factor=0.7)
        self.add(axes)

        # Wave definition functions
        def get_wave(n):
            return axes.plot(lambda x: sum([ (4/np.pi) * (np.sin((2*k-1)*x) / (2*k-1)) for k in range(1, n+1)]), color=color_1)

        wave = get_wave(1)
        self.add(wave)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(color_1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(color_2)
        wave_3 = get_wave(3)
        wave_3.set_color(color_2)
        self.play(Transform(wave, wave_3), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(color_3)
        wave_10 = get_wave(10)
        wave_10.set_color(color_3)
        self.play(Transform(wave, wave_10), run_time=2)
        self.wait(2)
