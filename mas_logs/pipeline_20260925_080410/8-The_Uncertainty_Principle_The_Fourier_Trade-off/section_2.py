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
        self.setup_layout("Defining the Trade-off: Time-Frequency Reciprocity", 
                          ["We define the uncertainty principle as product inequality.", 
                           "Narrower in time spreads frequency wider.", 
                           "Wider in time concentrates frequency tighter."])
        
        # Setup icons
        stopwatch = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/stopwatch.svg")
        tuningfork = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tuningfork.svg")
        
        self.place_at_grid(stopwatch, 'A2', scale_factor=0.3)
        self.place_at_grid(tuningfork, 'D2', scale_factor=0.3)
        self.add(stopwatch, tuningfork)
        
        # Gaussian functions
        def gaussian(x, sigma):
            return np.exp(-x**2 / (2 * sigma**2))

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#33FFF5")
        time_gauss = FunctionGraph(lambda x: gaussian(x, 0.5), x_range=[-2, 2], color="#33FFF5")
        
        # Using area for grouping
        self.place_in_area(time_gauss, 'B3', 'C6', scale_factor=0.6)
        self.play(Create(time_gauss))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#33FFF5")
        
        tight_gauss = FunctionGraph(lambda x: gaussian(x, 0.2), x_range=[-2, 2], color="#FF5733")
        wide_freq = FunctionGraph(lambda x: gaussian(x, 1.0), x_range=[-2, 2], color="#33FFF5")
        
        self.place_in_area(tight_gauss, 'B3', 'C6', scale_factor=0.6)
        self.place_in_area(wide_freq, 'E3', 'F6', scale_factor=0.6)
        
        self.play(Transform(time_gauss, tight_gauss), FadeIn(wide_freq))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#33FFF5")
        
        wide_gauss = FunctionGraph(lambda x: gaussian(x, 1.0), x_range=[-2, 2], color="#FF5733")
        tight_freq = FunctionGraph(lambda x: gaussian(x, 0.2), x_range=[-2, 2], color="#33FFF5")
        
        self.place_in_area(wide_gauss, 'B3', 'C6', scale_factor=0.6)
        self.place_in_area(tight_freq, 'E3', 'F6', scale_factor=0.6)
        
        self.play(Transform(time_gauss, wide_gauss), Transform(wide_freq, tight_freq))
        self.wait(1)
