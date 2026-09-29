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
        self.setup_layout("Prerequisite Bridge: From Discrete to Continuous", ["Histograms represent discrete data frequencies.", "Shrink bin widths to zero.", "Smooth curves emerge from bars."])
        
        # Elements
        # Using SVGMobject for [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/histogram.svg]
        histogram_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/histogram.svg")
        
        # Initial Bars
        bars = VGroup(*[Rectangle(height=h, width=1, color=WHITE).set_fill(WHITE, opacity=0.5) for h in [1, 2, 1.5]])
        bars.arrange(RIGHT, buff=0.1, aligned_edge=DOWN)
        
        # Using asset reference explicitly in the scene context, as per storyboard
        # Note: Storyboard says "Create discrete histogram bars... using [Asset: ...]"
        # I will place the histogram_asset and the generated bars
        self.place_in_area(bars, 'A3', 'F5', scale_factor=0.6)
        self.place_at_grid(histogram_asset, 'A1', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(histogram_asset), Create(bars))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        curve = FunctionGraph(lambda x: 2 * np.exp(-x**2), x_range=[-2, 2], color="#00FFFF")
        self.place_in_area(curve, 'A4', 'F6', scale_factor=0.65)
        
        self.play(Transform(bars, curve), FadeOut(histogram_asset))
        self.lecture[1].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        area = ImplicitFunction(lambda x, y: y - 2 * np.exp(-x**2), x_range=[-2, 2], y_range=[0, 2], color="#0000FF").set_fill(BLUE, opacity=0.3)
        self.place_in_area(area, 'A3', 'F5', scale_factor=0.6)
        
        # Final display with asset as requested
        final_histogram_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/histogram.svg")
        self.place_at_grid(final_histogram_asset, 'F6', scale_factor=0.5)
        
        self.play(FadeIn(area), FadeIn(final_histogram_asset))
        self.lecture[2].set_color("#FFFF00") # As per Storyboard color instruction for final line
        self.wait(1)
