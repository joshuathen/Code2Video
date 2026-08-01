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
        # Data from shared state
        title_text = "The Evolution: From Histogram to Curve"
        lecture_lines = [
            "Histograms show data frequency using discrete bars.",
            "Narrower bins create a smoother silhouette of data.",
            "Infinitely thin bins reveal the continuous PDF curve."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Helper function for the shape (Normal-like Distribution)
        def pdf_func(x):
            # Normal distribution shape for demonstration
            return 2.5 * np.exp(-0.5 * (x / 1.0)**2)

        # Global assets/elements
        baseline = Line(start=[-2.5, 0, 0], end=[2.5, 0, 0], color=WHITE)
        
        # Labels
        rainfall_label = Text("Rainfall Amount", font_size=18, color=WHITE)
        # Fix for Issue 26 & 27: Centering under chart and scaling down
        self.place_in_area(rainfall_label, 'F2', 'F6', scale_factor=0.6)
        
        freq_label = Text("Frequency", font_size=18, color=WHITE).rotate(90 * DEGREES)
        # Fix for Issue 25 & 27: Centering left of chart and scaling down
        self.place_in_area(freq_label, 'B1', 'E1', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#87CEFA")
        
        # Create 12-bar histogram
        num_bars_1 = 12
        x_range_1 = np.linspace(-2.2, 2.2, num_bars_1)
        width_1 = (x_range_1[1] - x_range_1[0]) * 0.8
        bars_1 = VGroup()
        for x in x_range_1:
            h = pdf_func(x)
            bar = Rectangle(
                width=width_1, 
                height=h, 
                fill_color="#87CEFA", 
                fill_opacity=0.8, 
                stroke_color="#87CEFA", 
                stroke_width=1
            )
            bar.move_to([x, h/2, 0])
            bars_1.add(bar)
            
        chart_1 = VGroup(bars_1, baseline)
        self.place_in_area(chart_1, "B2", "E6", scale_factor=0.7)
        
        self.play(
            Create(bars_1),
            Create(baseline),
            Write(rainfall_label),
            Write(freq_label),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00BFFF")
        
        # Create 365 thin bars (as per storyboard)
        num_bars_2 = 365
        x_range_2 = np.linspace(-2.2, 2.2, num_bars_2)
        width_2 = (x_range_2[1] - x_range_2[0])
        bars_2 = VGroup()
        for x in x_range_2:
            h = pdf_func(x)
            bar = Rectangle(
                width=width_2, 
                height=h, 
                fill_color="#00BFFF", 
                fill_opacity=0.8, 
                stroke_width=0
            )
            bar.move_to([x, h/2, 0])
            bars_2.add(bar)
        
        # Position chart_2 container identically
        chart_2_container = VGroup(bars_2, baseline.copy())
        self.place_in_area(chart_2_container, "B2", "E6", scale_factor=0.7)
        
        self.play(
            ReplacementTransform(bars_1, bars_2),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(WHITE)
        
        # Create smooth curve
        curve = ParametricFunction(
            lambda t: np.array([t, pdf_func(t), 0]),
            t_range=[-2.2, 2.2],
            color=WHITE,
            stroke_width=4
        )
        
        # Position curve container identically
        chart_3_container = VGroup(curve, baseline.copy())
        self.place_in_area(chart_3_container, "B2", "E6", scale_factor=0.7)
        
        self.play(Create(curve), run_time=2)
        self.play(bars_2.animate.set_fill(opacity=0), run_time=1)
        self.wait(3)

        # Final cleanup
        self.wait(1)
