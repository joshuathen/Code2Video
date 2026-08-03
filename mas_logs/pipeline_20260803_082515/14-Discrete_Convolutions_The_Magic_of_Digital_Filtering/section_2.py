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
        self.setup_layout(
            "Prerequisite: Discrete Signals and Arrays",
            [
                "A discrete signal is an ordered sequence of numbers.",
                "1D signals represent sound levels or sensor data.",
                "2D signals represent pixel grids in digital images."
            ]
        )

        # Define Colors
        color_1d_bars = "#ADD8E6"  # Light Blue
        color_1d_grid = "#F0E68C"  # Khaki
        color_2d_grid = "#FFFFFF"  # White

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(color_1d_bars)
        
        # Load Microphone Asset
        mic_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microphone.svg")
        mic_icon.set_color(color_1d_bars)
        self.place_at_grid(mic_icon, "A4", scale_factor=0.6)

        # Create a bar chart with heights [0, 1, 2, 1, 0] in light blue #ADD8E6
        # heights for visualization (0.05 for '0' to show the location)
        heights = [0.05, 1.0, 2.0, 1.0, 0.05]
        bars = VGroup(*[
            Rectangle(
                height=h, 
                width=0.6, 
                fill_opacity=0.8, 
                fill_color=color_1d_bars, 
                stroke_color=color_1d_bars
            )
            for h in heights
        ]).arrange(RIGHT, buff=0.2, aligned_edge=DOWN)
        
        # Baseline for the bar chart
        baseline = Line(
            start=bars.get_left() + LEFT*0.2, 
            end=bars.get_right() + RIGHT*0.2, 
            color=WHITE, 
            stroke_width=2
        ).next_to(bars, DOWN, buff=0)
        
        bar_chart_group = VGroup(bars, baseline)
        # Fix Issue 27: Problem: bar_chart_group is positioned at B2-D5, making it too close to the lecture text.
        # Fix: Line 88: self.place_in_area(bar_chart_group, 'B3', 'D6', scale_factor=1.0)
        self.place_in_area(bar_chart_group, 'B3', 'D6', scale_factor=1.0)
        
        self.play(FadeIn(mic_icon), Create(bar_chart_group), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(color_1d_grid)
        
        # Fade out bars and microphone, fade in a 1x5 grid of numbers.
        values = [0, 1, 2, 1, 0]
        grid_1d = VGroup()
        for val in values:
            sq = Square(side_length=0.8, stroke_color=WHITE)
            num = Text(str(val), font_size=24, color=color_1d_grid)
            grid_1d.add(VGroup(sq, num))
        grid_1d.arrange(RIGHT, buff=0.1)
        
        # Fix Issue 26: Problem: grid_1d starts at C1, which is too close to the left-side lecture notes.
        # Fix: Line 105: self.place_in_area(grid_1d, 'C2', 'C6', scale_factor=0.9)
        self.place_in_area(grid_1d, 'C2', 'C6', scale_factor=0.9)
        
        self.play(
            FadeOut(bar_chart_group),
            FadeOut(mic_icon),
            FadeIn(grid_1d)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(color_2d_grid)
        
        # Load Camera Asset
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        camera_icon.set_color(color_2d_grid)
        self.place_at_grid(camera_icon, "A4", scale_factor=0.6)

        # Show a 4x4 grid #FFFFFF representing a 2D image array.
        grid_2d = VGroup()
        for i in range(4):
            row = VGroup(*[
                Square(side_length=0.7, stroke_color=color_2d_grid, fill_opacity=0.1, fill_color=color_2d_grid) 
                for _ in range(4)
            ]).arrange(RIGHT, buff=0.05)
            grid_2d.add(row)
        grid_2d.arrange(DOWN, buff=0.05)
        
        # Fix Issue 28: Problem: grid_2d is positioned at B2-E5, causing visual imbalance.
        # Fix: Line 126: self.place_in_area(grid_2d, 'B3', 'E6', scale_factor=1.0)
        self.place_in_area(grid_2d, 'B3', 'E6', scale_factor=1.0)
        
        self.play(
            FadeOut(grid_1d),
            FadeIn(camera_icon),
            FadeIn(grid_2d)
        )
        self.wait(2)
