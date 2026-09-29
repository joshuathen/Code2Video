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
            "Dispersion occurs when frequency nears resonance.",
            "Different frequencies shift phase by varying amounts.",
            "Thus, refractive index depends on light's frequency."
        ]
        self.setup_layout("Color Dependence: Dispersion", lecture_lines)
        
        # Assets
        bulb = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bulb.svg")
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")
        
        # Red wave (Low freq)
        red_wave = FunctionGraph(lambda x: 0.5 * np.sin(2 * x), x_range=[-2, 2], color=RED)
        self.place_in_area(red_wave, 'A2', 'B5', scale_factor=0.6)
        
        # Blue wave (High freq)
        blue_wave = FunctionGraph(lambda x: 0.5 * np.sin(4 * x), x_range=[-2, 2], color=BLUE)
        self.place_in_area(blue_wave, 'D2', 'E5', scale_factor=0.6)

        legend_label = Text("n(ω)", font_size=24)
        self.place_at_grid(legend_label, 'C3', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(bulb, 'A1', scale_factor=0.4)
        self.play(FadeIn(bulb), Create(red_wave))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.play(Create(blue_wave))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(prism, 'C6', scale_factor=0.4)
        self.play(Indicate(red_wave), Indicate(blue_wave), FadeIn(prism))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
