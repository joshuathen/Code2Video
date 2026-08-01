import os
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

class Section6Scene(TeachingScene):
    def construct(self):
        # Define the lecture lines for layout
        lecture_lines = [
            'Statistics helps us predict rare events in the real-world.',
            'Imagine a hatchery where rare gold dragons are five percent.',
            'Binomial math calculates the odds for any specific clutch.'
        ]
        
        self.setup_layout("Real-World Application: Quality Control", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Colors: Line 1 (White)
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Create a grid of 20 egg icons [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/egg.svg] (#FFFFFF)
        eggs = VGroup()
        rows_grid = ["A", "B", "C", "D"]
        cols_grid = ["1", "2", "3", "4", "5"]
        
        for r_idx, row in enumerate(rows_grid):
            for c_idx, col in enumerate(cols_grid):
                # Using egg SVG asset
                egg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/egg.svg").set_color(WHITE)
                self.place_at_grid(egg, f"{row}{col}", scale_factor=0.35)
                eggs.add(egg)
        
        # Labels for parameters n and p
        n_label = Text("n = 20", font_size=32, color="#FFFFFF")
        p_label = Text("p = 0.05", font_size=32, color="#FFFFFF")
        self.place_at_grid(n_label, "A6")
        self.place_at_grid(p_label, "B6")
        
        self.play(
            FadeIn(eggs, shift=UP),
            Write(n_label),
            Write(p_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Colors: Line 2 (Yellow)
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        # Highlight 2 random eggs in gold (#FFFF00)
        # Using specific indices for consistency: 3 and 12
        egg_indices = [3, 12]
        highlight_animations = []
        for idx in egg_indices:
            highlight_animations.append(eggs[idx].animate.set_color("#FFFF00"))
            
        # Formula for P(X=2) (Placement fixed per Issue 37)
        formula = Text(
            "P(X=2) = 20C2 * (0.05)^2 * (0.95)^18",
            font_size=24,
            color="#FFFF00"
        )
        self.place_in_area(formula, "E2", "E6", scale_factor=0.7)
        
        self.play(
            *highlight_animations,
            Write(formula)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Colors: Line 3 (Cyan)
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        
        # Fade in the calculated result '0.1887' (Placement fixed per Issue 38)
        result_text = Text("approx. 0.1887", font_size=36, color="#00FFFF")
        self.place_in_area(result_text, "F2", "F4", scale_factor=0.7)
        
        # 'Rare Success' dragon icon [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/dragon.svg]
        # (Placement fixed per Issue 39)
        dragon_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/dragon.svg").set_color("#00FFFF")
        badge_text = Text("RARE SUCCESS", font_size=18, color="#00FFFF")
        badge = VGroup(dragon_icon, badge_text).arrange(DOWN, buff=0.1)
        self.place_in_area(badge, "F5", "F6", scale_factor=0.7)
        
        self.play(
            FadeIn(result_text, shift=RIGHT),
            FadeIn(badge, shift=UP)
        )
        self.wait(3)
