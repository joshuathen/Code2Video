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

class Section3Scene(TeachingScene):
    def construct(self):
        # Initial layout setup
        lecture_lines = [
            'We repeat this sampling process many, many times.',
            'Watch as thousands of sample averages are plotted.',
            'A specific shape begins to emerge from the chaos.',
            'Despite the messy start, we see a bell curve.',
            'This is the magic of the sampling distribution.'
        ]
        self.setup_layout("The Transformation (The 'Magic' Moment)", lecture_lines)

        # Define Colors
        DOT_COLOR = "#FFFF00"
        BELL_COLOR = "#FFFFFF"
        MEAN_COLOR = "#00FFFF"
        MESSY_COLOR = "#FFA500"
        LABEL_COLOR = "#FFD700"

        # Setup Axes for the simulation
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 1, 0.2],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": False, "color": GRAY}
        )
        self.place_in_area(axes, "B1", "F6", scale_factor=0.8)

        # Normal Distribution Curve for Line 3 onwards
        def gaussian(x):
            return np.exp(-x**2 / 1.0) * 0.8
        
        bell_curve = axes.plot(gaussian, color=BELL_COLOR, stroke_width=4)
        
        # Messy Curve for Line 4 comparison
        def messy_distribution(x):
            return (0.3 * np.exp(-(x+1.5)**2 / 0.5) + 0.5 * np.exp(-(x-1.0)**2 / 0.8)) * 0.7
        
        messy_curve = axes.plot(messy_distribution, color=MESSY_COLOR, stroke_width=3)

        # === Animation for Lecture Line 1 ===
        # Line 1 is active (White is default)
        sample_indicator = Text("Sampling...", font_size=20, color=WHITE)
        # Fix Issue 45: Reposition indicator to avoid overlap with growing bars
        self.place_at_grid(sample_indicator, "A4", scale_factor=0.8)
        self.play(Write(sample_indicator))
        self.wait(1)
        self.play(FadeOut(sample_indicator))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(DOT_COLOR)
        self.add(axes)

        # Animation: Dots stacking (Simulating thousands)
        num_dots = 100 
        dots = VGroup()
        bins = {i: 0 for i in np.arange(-3, 3.1, 0.2)}
        
        for _ in range(num_dots):
            val = np.random.normal(0, 0.7)
            closest_bin = min(bins.keys(), key=lambda x: abs(x - val))
            bins[closest_bin] += 0.05
            dot = Dot(point=axes.c2p(closest_bin, bins[closest_bin]), radius=0.04, color=DOT_COLOR)
            dots.add(dot)

        self.play(LaggedStart(*(Create(d) for d in dots), lag_ratio=0.015, run_time=3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(MEAN_COLOR)

        # Transition dots to curve
        self.play(FadeOut(dots), Create(bell_curve), run_time=1.5)

        # Vertical line of symmetry
        mean_line = axes.get_vertical_line(axes.c2p(0, gaussian(0)), color=MEAN_COLOR, line_func=DashedLine)
        mean_label = Text("Mean of Averages", font_size=16, color=MEAN_COLOR)
        # Position label in Row A to avoid cluttering axes center
        self.place_in_area(mean_label, "A3", "A4", scale_factor=0.8)

        self.play(Create(mean_line), Write(mean_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(MESSY_COLOR)

        messy_label = Text("Original Shape", font_size=16, color=MESSY_COLOR)
        # Fix Issue 46: Move label to top area to avoid cluttering plot area
        self.place_in_area(messy_label, "A1", "A2", scale_factor=0.8)
        
        self.play(Create(messy_curve), Write(messy_label))
        self.play(Indicate(messy_curve, color=MESSY_COLOR), Indicate(bell_curve, color=BELL_COLOR))
        self.wait(2)
        self.play(FadeOut(messy_curve), FadeOut(messy_label))

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(LABEL_COLOR)

        normal_label = Text("Normal Distribution", font_size=24, color=LABEL_COLOR)
        # Fix Issue 44: Reposition label to avoid overlapping graph tail
        self.place_in_area(normal_label, "A5", "A6", scale_factor=0.8)
        
        self.play(Write(normal_label))
        self.play(normal_label.animate.scale(1.1), run_time=0.5, rate_func=there_and_back)
        self.wait(2)

        # Reset colors for final state
        self.lecture[4].set_color(WHITE)
        self.wait(1)
