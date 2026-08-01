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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Imagine Sam the Squirrel collecting nuts every day.",
            "He follows a pattern: one, then one-fourth, then one-ninth.",
            "Does this infinite pile grow forever?",
            "Surprisingly, the total nuts stay under a finite limit.",
            "This reveals the beauty of converging infinite sums."
        ]
        self.setup_layout("The Hook: The Infinite Nut Collector", lecture_lines)

        # Colors
        BROWN = "#8B4513"
        GREEN = "#00FF00"
        RED = "#FF0000"
        GOLD = "#FFD700"
        LIGHT_BLUE = "#ADD8E6"

        # === Animation for Lecture Line 1 ===
        # Fade in Sam [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/squirrel.svg]
        sam = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/squirrel.svg")
        sam.set_color(BROWN)
        self.place_at_grid(sam, "A1", scale_factor=0.6)
        
        series_label = Text("Infinite Series", color=WHITE, font_size=24)
        self.place_at_grid(series_label, "A3", scale_factor=0.8)
        
        self.play(self.lecture[0].animate.set_color(BROWN))
        self.play(FadeIn(sam), Write(series_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Bars stack up for 1 + 1/4 + 1/9... (#00FF00)
        self.play(self.lecture[1].animate.set_color(GREEN))
        
        basel_bars = VGroup()
        for i in range(1, 6):
            h = (1 / (i**2)) * 1.5
            bar = Rectangle(width=0.4, height=h, fill_opacity=0.8, fill_color=GREEN, stroke_color=GREEN)
            basel_bars.add(bar)
        basel_bars.arrange(RIGHT, buff=0.1, aligned_edge=DOWN)
        
        # Addressing spirit of Issue 22: anchoring bars in the upper area
        self.place_in_area(basel_bars, "B2", "D4", scale_factor=1.0)
        
        self.play(Create(basel_bars))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Contrast with harmonic bars (#FF0000)
        self.play(self.lecture[2].animate.set_color(RED))
        
        harmonic_bars = VGroup()
        for i in range(1, 6):
            h = (1 / i) * 1.5
            bar = Rectangle(width=0.4, height=h, fill_opacity=0.5, fill_color=RED, stroke_color=RED)
            harmonic_bars.add(bar)
        harmonic_bars.arrange(RIGHT, buff=0.1, aligned_edge=DOWN)
        
        # Addressing spirit of Issue 23: anchoring contrast bars in the lower area
        self.place_in_area(harmonic_bars, "D2", "F4", scale_factor=1.0)
        
        self.play(Create(harmonic_bars))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Limit line at pi^2/6, marked by a nut [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/nut.svg]
        self.play(self.lecture[3].animate.set_color(GOLD))
        
        # Calculate limit line relative to basel_bars
        bb_bottom = basel_bars.get_bottom()[1]
        limit_h = (np.pi**2 / 6) * 1.5
        
        limit_line = Line(
            start=[self.grid["B1"][0], bb_bottom + limit_h, 0],
            end=[self.grid["B6"][0], bb_bottom + limit_h, 0],
            color=GOLD, stroke_width=3
        )
        
        nut = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/nut.svg")
        nut.set_color(GOLD)
        # Addressing spirit of Issue 24: anchoring the marker on the grid
        self.place_at_grid(nut, "B5", scale_factor=0.3)
        # Align nut vertically with the limit line
        nut.set_y(bb_bottom + limit_h)
        
        self.play(Create(limit_line), FadeIn(nut))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(LIGHT_BLUE))
        self.play(Indicate(basel_bars), Flash(nut, color=GOLD))
        self.wait(2)
