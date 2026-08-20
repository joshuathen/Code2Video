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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Real-World Application & Summary", [
            "Fourier transforms time-domain to frequency-domain data.",
            "Spectrum analyzers visualize these frequency components.",
            "Compression algorithms use this to remove data."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Waveform
        wave = FunctionGraph(lambda x: 0.3 * np.sin(4 * x), x_range=[-2, 2], color=WHITE)
        self.place_in_area(wave, 'A1', 'A6', scale_factor=0.6)
        self.play(Create(wave))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        # Spectrum analyzer with asset
        analyzer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/analyzer.svg", color="#FF00FF")
        self.place_at_grid(analyzer_icon, 'C2', scale_factor=0.4)
        
        bars = VGroup(*[Rectangle(height=0.5, width=0.2, color="#FF00FF", fill_opacity=0.7) for _ in range(5)])
        bars.arrange(RIGHT, buff=0.1)
        self.place_at_grid(bars, 'D2', scale_factor=0.6)
        
        # Simple animation for bars to avoid heavy updaters if possible
        self.play(FadeIn(analyzer_icon), AnimationGroup(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.1))
        
        self.lecture[1].set_color("#FF00FF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Compression illustration
        rect = Rectangle(width=2, height=1, color="#00FFFF")
        text = Text("Data", font_size=24, color="#00FFFF")
        comp = VGroup(rect, text)
        self.place_at_grid(comp, 'E4', scale_factor=0.8)
        
        self.play(FadeIn(comp))
        self.play(comp.animate.scale(0.5))
        
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
