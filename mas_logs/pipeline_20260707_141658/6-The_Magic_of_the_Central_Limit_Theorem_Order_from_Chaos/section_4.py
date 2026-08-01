from manim import *
import numpy as np

# Use the provided TeachingScene base class without modification.
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
        # Initialize layout
        self.setup_layout("The Convergence: The Birth of the Bell", [
            "- Watch as sample means begin to pile up.",
            "- A symmetric bell shape naturally starts to form.",
            "- This is the birth of the Normal Distribution."
        ])
        
        # Define hex colors
        BLUE_HEX = "#0000FF"
        WHITE_HEX = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Lecture: "- Watch as sample means begin to pile up."
        self.play(self.lecture[0].animate.set_color(BLUE_HEX))
        
        # Create an axis for the histogram
        axis = Line(self.grid["E1"], self.grid["E6"], color=WHITE)
        axis_label = Text("Sample Means", font_size=20, color=WHITE)
        # Fix for Issue 44: Center label below axis using area F3-F4
        self.place_in_area(axis_label, "F3", "F4", scale_factor=0.8)
        
        # Create histogram bars
        num_bars = 12
        bar_width = 0.35
        bars = VGroup()
        
        # Start bars at zero height
        for i in range(num_bars):
            bar = Rectangle(
                width=bar_width,
                height=0.01,
                fill_color=BLUE_HEX,
                fill_opacity=0.7,
                stroke_width=1,
                stroke_color=BLUE_HEX
            )
            # Distribute bars across columns 1 to 6 on row E
            x_pos = self.grid["E1"][0] + i * (self.grid["E6"][0] - self.grid["E1"][0]) / (num_bars - 1)
            bar.move_to([x_pos, self.grid["E1"][1], 0], aligned_edge=DOWN)
            bars.add(bar)
        
        self.add(axis, axis_label, bars)
        
        # Initial jagged piling animation
        jagged_heights = [0.2, 0.8, 1.5, 0.4, 2.2, 1.1, 2.5, 0.9, 1.8, 0.5, 1.2, 0.3]
        self.play(
            *[bars[i].animate.stretch_to_fit_height(jagged_heights[i], about_edge=DOWN) for i in range(num_bars)],
            run_time=2.0
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Lecture: "- A symmetric bell shape naturally starts to form."
        self.play(self.lecture[1].animate.set_color(BLUE_HEX))
        
        # Symmetric heights for bars (Gaussian-like)
        def gaussian(x, mu, sig):
            return 3.5 * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        
        smooth_heights = [gaussian(i, (num_bars-1)/2, 2.0) for i in range(num_bars)]
        
        self.play(
            *[bars[i].animate.stretch_to_fit_height(smooth_heights[i], about_edge=DOWN) for i in range(num_bars)],
            run_time=2.0
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Lecture: "- This is the birth of the Normal Distribution."
        self.play(self.lecture[2].animate.set_color(WHITE_HEX))
        
        # Define the bell curve shape
        bell_curve = FunctionGraph(
            lambda x: 3.5 * np.exp(-0.5 * (x / 1.0)**2),
            x_range=[-3, 3],
            color=WHITE_HEX,
            stroke_width=4
        )
        
        # Fix for Issue 43 & 45: Position and scale distribution elements
        # Create a group for the whole visual to ensure centering
        distribution_group = VGroup(bars, axis, bell_curve)
        self.place_in_area(distribution_group, "B2", "E5", scale_factor=1.1)
        
        # Since place_in_area moves the group, we need to ensure the curve is added with proper effects
        # The bell_curve is already in the group, so it's positioned.
        
        # Add glow effect to curve
        glow = bell_curve.copy().set_stroke(width=12, opacity=0.3, color=WHITE_HEX)
        
        # Animate the curve's appearance
        self.play(Create(bell_curve), FadeIn(glow), run_time=2)
        
        # Pulsing effect for the "Birth" (Issue 45 context)
        self.play(
            glow.animate.set_stroke(width=20, opacity=0.6),
            rate_func=there_and_back,
            run_time=1.0
        )
        
        self.wait(3)
