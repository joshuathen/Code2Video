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
        title = "Why It Matters: Real-World Application"
        lines = [
            "CLT helps us predict behavior in complex systems.",
            "Imagine a factory making bars of varying weights.",
            "Averages help managers ensure consistent quality control."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Conveyor belt base
        belt = Line(self.grid["B1"], self.grid["B6"], color=GREY_B, stroke_width=8)
        belt_label = Text("Cocoa-Bot 3000 Line", font_size=18, color=GREY_A)
        self.place_in_area(belt_label, "A2", "A5", scale_factor=0.8)
        
        self.play(Create(belt), Write(belt_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Create 5 chocolate bars
        bars = VGroup()
        for i in range(5):
            bar = Rectangle(width=0.6, height=0.3, fill_color="#A52A2A", fill_opacity=1, stroke_color=WHITE, stroke_width=1)
            # Stagger them slightly
            self.place_at_grid(bar, f"B{i+1}")
            bars.add(bar)
            
        # Move bars along conveyor
        self.play(
            LaggedStart(*[bar.animate.shift(RIGHT * 0.5) for bar in bars], lag_ratio=0.2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Grouping 40 bars (visualized as a cluster)
        sampling_box = DashedVMobject(Rectangle(width=2, height=1, color=BLUE))
        self.place_in_area(sampling_box, "B2", "C5")
        sampling_text = Text("Sample (n=40)", font_size=16, color=BLUE)
        sampling_text.next_to(sampling_box, UP, buff=0.1)

        # Hide old bars, show sampling
        self.play(
            FadeOut(bars),
            FadeIn(sampling_box),
            FadeIn(sampling_text)
        )
        
        # Bell curve axes (Manual creation to fit grid)
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 1, 0.5],
            axis_config={"include_tip": False, "stroke_width": 2},
            x_length=4,
            y_length=2
        )
        self.place_in_area(axes, "D1", "F6", scale_factor=0.9)
        
        # Gaussian curve (narrow sigma for averages)
        curve = axes.plot(lambda x: np.exp(-x**2 / (2 * 0.5**2)) / (0.5 * np.sqrt(2 * np.pi)), color=YELLOW)
        
        # Failure Zones (Tails)
        failure_zone_left = axes.get_area(curve, x_range=[-3, -1.2], color="#FF0000", opacity=0.5)
        failure_zone_right = axes.get_area(curve, x_range=[1.2, 3], color="#FF0000", opacity=0.5)
        
        fail_label = Text("Failure Zone", font_size=18, color="#FF0000")
        self.place_at_grid(fail_label, "E6", scale_factor=0.6)

        # Animation sequence
        self.play(
            Create(axes),
            Create(curve),
            FadeOut(sampling_box, sampling_text, belt, belt_label),
            run_time=2
        )
        
        # Dot representing an average falling onto the curve
        avg_dot = Dot(color=WHITE).move_to(axes.c2p(0, 0))
        self.play(avg_dot.animate.move_to(axes.c2p(0, 0.8)), run_time=1)
        
        self.play(
            FadeIn(failure_zone_left),
            FadeIn(failure_zone_right),
            Write(fail_label)
        )
        
        self.wait(3)
        self.lecture[2].set_color(WHITE)
