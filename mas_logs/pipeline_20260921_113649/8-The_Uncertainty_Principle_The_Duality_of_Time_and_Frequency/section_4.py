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
        self.setup_layout("Real-World Application: The Bat's Chirp", [
            "Bats balance precision via sonar.",
            "Short clicks improve timing resolution.",
            "Long whistles improve frequency resolution."
        ])
        
        # Load asset
        bat_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bat.svg")
        self.place_at_grid(bat_icon, 'A3', scale_factor=0.5)
        self.add(bat_icon)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        wave = FunctionGraph(lambda t: 0.5 * np.sin(5 * np.pi * t), x_range=[-2, 2]).set_color(WHITE)
        self.place_in_area(wave, 'B2', 'C4', scale_factor=0.6)
        self.play(Create(wave))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        # Short click: high freq, short duration
        click = FunctionGraph(lambda t: 0.5 * np.sin(20 * np.pi * t) * np.exp(-5*t**2), x_range=[-1, 1], color=GREEN)
        self.place_at_grid(click, 'D2', scale_factor=0.7)
        self.play(ReplacementTransform(wave, click))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        # Long whistle: low freq, long duration
        whistle = FunctionGraph(lambda t: 0.5 * np.sin(2 * np.pi * t) * np.exp(-0.5*t**2), x_range=[-2, 2], color=YELLOW)
        self.place_in_area(whistle, 'D4', 'E5', scale_factor=0.7)
        self.play(ReplacementTransform(click, whistle))
        self.wait(2)
