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
        self.setup_layout("Visualization: The Morphing Animation", [
            "Watch the histogram morph over time.", 
            "As samples increase, patterns stabilize.", 
            "Chaos collapses into symmetry."
        ])
        
        # Load assets
        dice = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg")
        sand = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sand.svg")
        
        # Initial Histogram
        bins = 15
        hist = VGroup(*[Rectangle(width=0.4, height=0.5, fill_opacity=0.8, color="#D35400", stroke_width=0) for _ in range(bins)])
        hist.arrange(RIGHT, buff=0.1, aligned_edge=DOWN)
        self.place_in_area(hist, 'B1', 'E6', scale_factor=0.5)
        
        # Pre-place icons
        self.place_at_grid(dice, 'A6', scale_factor=0.3)
        self.place_at_grid(sand, 'F6', scale_factor=0.3)
        sand.set_opacity(0) # Hide until needed
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(hist), FadeIn(dice), self.lecture[0].animate.set_color("#D35400"))
        
        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE), 
            self.lecture[1].animate.set_color("#7F8C8D"),
            *[bar.animate.set_color("#7F8C8D") for bar in hist],
            run_time=2.0
        )
        
        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE), 
            self.lecture[2].animate.set_color("#27AE60"), 
            FadeIn(sand)
        )
        
        # Simple bell shape
        for i, bar in enumerate(hist):
            x = i - (bins - 1) / 2
            h = 3.0 * np.exp(-0.15 * x**2)
            bar.generate_target()
            bar.target.set_height(h)
            bar.target.set_color("#27AE60")
        
        self.play(MoveToTarget(hist), run_time=2.0)
        self.wait(2)
