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
        self.setup_layout("The Problem: The 'Wild' Population", [
            "Populations are rarely perfectly bell-shaped.",
            "Some exhibit wild, irregular behaviors.",
            "What happens when we sample them?"
        ])
        
        # === Animation for Lecture Line 1 ===
        # Create irregular histogram
        bars = VGroup(*[
            Rectangle(width=0.8, height=h, fill_opacity=0.8, color="#FF5733", stroke_width=0)
            for h in [0.5, 1.2, 0.8, 1.8, 0.6, 1.5]
        ]).arrange(RIGHT, aligned_edge=DOWN, buff=0.1)
        self.place_in_area(bars, "B2", "D5")
        self.play(FadeIn(bars))
        self.lecture[0].set_color("#FF5733")

        # === Animation for Lecture Line 2 ===
        # Add outliers using asset
        outlier = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/outlier.svg")
        outlier.set_color("#FF3333")
        self.place_at_grid(outlier, "B5", scale_factor=0.5)
        self.play(FadeIn(outlier), run_time=1)
        self.lecture[1].set_color("#FF3333")

        # === Animation for Lecture Line 3 ===
        # Display question mark
        qm = Text("?", font_size=72, color=YELLOW)
        self.place_at_grid(qm, "A3")
        self.play(GrowFromCenter(qm))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
