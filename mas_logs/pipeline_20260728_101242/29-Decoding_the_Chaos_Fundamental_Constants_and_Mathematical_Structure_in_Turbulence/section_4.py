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
        self.setup_layout(
            "Kolmogorov’s K41 Theory and the -5/3 Law",
            [
                "Kolmogorov theorized that small-scale turbulence is statistically universal.",
                "Energy distribution follows the famous -5/3 power law.",
                "This law appears as a straight line on log-graphs.",
                "It relates energy to the wave number of the eddies.",
                "The Kolmogorov constant provides a universal mathematical signature."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Create a log-log axes simulation. X-axis: 'Wave Number (k)', Y-axis: 'Energy E(k)' (#FFFFFF).
        self.lecture[0].set_color(YELLOW)
        
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            x_length=3.5,
            y_length=3.5,
            axis_config={"include_tip": True, "color": WHITE},
            tips=False
        )
        # Issue 34 Fix: Shift axes further right.
        self.place_in_area(axes, "B3", "E6")
        
        x_label = Text("Wave Number (k)", font_size=16, color=WHITE)
        y_label = Text("Energy E(k)", font_size=16, color=WHITE).rotate(90 * DEGREES)
        
        # Issue 36 Fix: Re-center x-label under shifted axes.
        self.place_at_grid(x_label, "F4", scale_factor=1.0)
        # Issue 34 Fix: Shift y-label to D2.
        self.place_at_grid(y_label, "D2", scale_factor=1.0)
        
        self.play(Create(axes), Write(x_label), Write(y_label), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A line plots from top-left to bottom-right across the graph (#ADD8E6).
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        start_point = axes.c2p(1, 9)
        end_point = axes.c2p(9, 1)
        full_line = Line(start_point, end_point, color="#ADD8E6", stroke_width=4)
        
        self.play(Create(full_line), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight the middle section of the line in yellow (#FFFF00). Label: 'Inertial Subrange' (#FFFFFF).
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        mid_proportion_start = 0.2
        mid_proportion_end = 0.8
        mid_start = full_line.point_from_proportion(mid_proportion_start)
        mid_end = full_line.point_from_proportion(mid_proportion_end)
        highlight_line = Line(mid_start, mid_end, color="#FFFF00", stroke_width=6)
        
        subrange_label = Text("Inertial Subrange", font_size=18, color=WHITE)
        # Issue 35 Fix: Move label away from trend line to B6.
        self.place_at_grid(subrange_label, "B6", scale_factor=1.0)
        
        self.play(
            Create(highlight_line),
            Write(subrange_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Display the equation E(k) ~ k^(-5/3) next to the highlighted section (#FFFFFF).
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        equation = MathTex("E(k) \\propto k^{-5/3}", font_size=32, color=WHITE)
        # Issue 35 Fix: Move equation away from trend line to C6.
        self.place_at_grid(equation, "C6", scale_factor=1.2)
        
        self.play(Write(equation), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The text '-5/3' flashes and becomes bold (#FFFF00) to emphasize the universal slope.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Kolmogorov constant is context for the formula.
        # Target symbols '-5/3' in "E(k) \propto k^{-5/3}"
        # Symbols: E, (, k, ), \propto, k, -, 5, /, 3
        exponent = equation[0][6:10]
        
        self.play(
            exponent.animate.set_color("#FFFF00").scale(1.2),
            Flash(exponent, color="#FFFF00", line_length=0.2),
            run_time=1.0
        )
        self.wait(2)
        
        self.lecture[4].set_color(WHITE)
        self.wait(1)
