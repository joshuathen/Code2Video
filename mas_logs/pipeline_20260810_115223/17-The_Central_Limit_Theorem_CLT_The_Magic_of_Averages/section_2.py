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
        lecture_lines = [
            "The CLT bridges chaos and perfect order.",
            "Population shapes can be wild and messy.",
            "Yet, sample means aggregate into normal distributions.",
            "As samples grow, the bell curve emerges.",
            "This paradox is the magic of statistics."
        ]
        self.setup_layout("The CLT Paradox: From Chaos to Order", lecture_lines)
        
        # Assets
        hist_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/histogram.svg")
        sample_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sample.svg")
        bell_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bell.svg")
        
        hist_icon.set_color("#FF5733")
        bell_icon.set_color("#33FF57")
        
        # Initial positions - Updated based on VideoCritic requirements
        self.place_in_area(hist_icon, "C2", "E4", scale_factor=0.9)
        self.place_at_grid(sample_icon, "B3", scale_factor=0.6)
        self.place_at_grid(bell_icon, "C4", scale_factor=0.7)
        
        # Ensure icons start hidden for fadeIn sequence
        hist_icon.set_opacity(0)
        sample_icon.set_opacity(0)
        bell_icon.set_opacity(0)
        
        # Add to scene
        self.add(hist_icon, sample_icon, bell_icon)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5733"), hist_icon.animate.set_opacity(1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"), sample_icon.animate.set_opacity(1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"), sample_icon.animate.set_opacity(0))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFF00"), bell_icon.animate.set_opacity(1))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FFFF"))
        self.wait(1)
