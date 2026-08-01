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
        # Initial Setup
        title = "Key Metrics: Mean and Variance"
        lines = [
            "We can predict the average outcome using the mean.",
            "Multiplying trials by success probability gives the expected value.",
            "Standard deviation measures the typical spread around that average."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Display formulas Mean = n*p using Text to avoid LaTeX dependency
        mean_formula = Text("Mean = n × p", color=WHITE)
        self.place_in_area(mean_formula, 'B3', 'B6', scale_factor=0.9)
        
        self.play(Write(mean_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Display SD formula using Text to avoid LaTeX dependency
        sd_formula = Text("SD = √(n × p × (1 - p))", color=WHITE)
        self.place_in_area(sd_formula, 'C3', 'C6', scale_factor=0.8)
        
        # Setup Gauge (n=100, p=0.7 means expected value is 70)
        gauge = NumberLine(
            x_range=[0, 100, 10],
            length=5,
            include_numbers=True,
            label_constructor=Text,
            font_size=16,
            color=WHITE
        )
        self.place_in_area(gauge, "E1", "E6", scale_factor=0.9)
        
        gauge_label = Text("Predicted Score Meter (n=100, p=0.7)", font_size=18, color=WHITE)
        self.place_in_area(gauge_label, 'D1', 'D6', scale_factor=0.6)
        
        needle = Arrow(
            start=UP * 0.6,
            end=ORIGIN,
            color="#FFFF00",
            buff=0,
            stroke_width=6
        )
        
        # Position needle at 0 initially
        needle.move_to(gauge.number_to_point(0) + UP * 0.3)
        
        self.play(
            Write(sd_formula),
            Create(gauge),
            FadeIn(gauge_label),
            FadeIn(needle)
        )
        
        # Animate needle to 70 (Mean = 100 * 0.7)
        self.play(needle.animate.move_to(gauge.number_to_point(70) + UP * 0.3), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Standard deviation region
        # SD = sqrt(100 * 0.7 * 0.3) = sqrt(21) approx 4.58
        sd_value = np.sqrt(100 * 0.7 * 0.3)
        left_bound = gauge.number_to_point(70 - sd_value)
        right_bound = gauge.number_to_point(70 + sd_value)
        
        sd_region = Rectangle(
            width=right_bound[0] - left_bound[0],
            height=0.4,
            fill_color="#83C167",
            fill_opacity=0.4,
            stroke_width=0
        )
        sd_region.move_to(gauge.number_to_point(70))
        
        sd_label = Text("Standard Deviation Spread", font_size=14, color="#83C167")
        self.place_in_area(sd_label, 'F1', 'F6', scale_factor=0.7)
        
        self.play(
            FadeIn(sd_region, shift=DOWN),
            FadeIn(sd_label)
        )
        
        # Final highlight
        self.wait(2)
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
