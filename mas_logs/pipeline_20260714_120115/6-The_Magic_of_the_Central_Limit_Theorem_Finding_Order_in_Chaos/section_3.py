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
        # Data from Storyboard
        title = "The Transformation: Emergence of the Bell Curve"
        lecture_lines = [
            "Observe how the population's starting shape is extremely jagged.",
            "But the collection of sample averages forms a pattern.",
            "It settles into a smooth, symmetric bell-shaped curve.",
            "The same shape appears regardless of the starting distribution.",
            "This transformation is the heart of the Central Limit Theorem."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        COLOR_TOP = "#FFFFFF"
        COLOR_HIST = "#FF8080"
        COLOR_BELL = "#00FF00"
        COLOR_LINE = "#FFFF00"
        COLOR_CLT = "#FFFFFF"

        # Setup Axes for Top Plot (Population)
        top_axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 5, 1],
            x_length=4.5,
            y_length=2,
            axis_config={"include_tip": False, "color": WHITE}
        )
        self.place_in_area(top_axes, 'B2', 'C6', scale_factor=0.8)

        # Setup Axes for Bottom Plot (Sample Means)
        bottom_axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 1.2, 0.2],
            x_length=4.5,
            y_length=2.5,
            axis_config={"include_tip": False, "color": WHITE}
        )
        self.place_in_area(bottom_axes, 'D2', 'F6', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # Observe how the population's starting shape is extremely jagged.
        self.lecture[0].set_color(COLOR_TOP)
        
        def bimodal_func(x):
            return 4.5 * (np.exp(-(x-2.5)**2 / 1.5) + np.exp(-(x-7.5)**2 / 1.5)) / 1.2
        
        bimodal_plot = top_axes.plot(bimodal_func, color=COLOR_TOP)
        top_label = Text("Original Population", font_size=20, color=COLOR_TOP)
        # Fix from VideoCritic: Move to A4, scale 0.6
        self.place_at_grid(top_label, 'A4', scale_factor=0.6)

        self.play(Create(top_axes), Create(bimodal_plot), FadeIn(top_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # But the collection of sample averages forms a pattern.
        self.lecture[1].set_color(COLOR_HIST)
        
        # Create a "histogram" using bars
        bars = VGroup()
        num_bars = 25
        for i in range(num_bars):
            x_val = -2.4 + i * (4.8 / (num_bars - 1))
            # Theoretical Normal Distribution heights
            h_val = np.exp(-x_val**2 / 1.2)
            bar = Rectangle(
                width=0.12,
                height=h_val * bottom_axes.y_axis.unit_size,
                fill_color=COLOR_HIST,
                fill_opacity=0.7,
                stroke_width=0.5,
                stroke_color=COLOR_HIST
            )
            bar.move_to(bottom_axes.c2p(x_val, 0), aligned_edge=DOWN)
            bars.add(bar)
        
        bottom_label = Text("Sample Means Distribution", font_size=20, color=COLOR_HIST)
        # Fix from VideoCritic: Move to C4, scale 0.6
        self.place_at_grid(bottom_label, 'C4', scale_factor=0.6)

        self.play(Create(bottom_axes), FadeIn(bottom_label), LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.03))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # It settles into a smooth, symmetric bell-shaped curve.
        self.lecture[2].set_color(COLOR_BELL)
        
        bell_curve = bottom_axes.plot(lambda x: np.exp(-x**2 / 1.2), color=COLOR_BELL, stroke_width=4)
        bell_glow = bell_curve.copy().set_stroke(width=10, opacity=0.3)
        
        self.play(Create(bell_curve), FadeIn(bell_glow))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The same shape appears regardless of the starting distribution.
        self.lecture[3].set_color(COLOR_TOP)
        
        uniform_plot = top_axes.plot(lambda x: 2.5, color=COLOR_TOP)
        uniform_label = Text("Uniform Population", font_size=20, color=COLOR_TOP)
        self.place_at_grid(uniform_label, 'A4', scale_factor=0.6)

        self.play(
            ReplacementTransform(bimodal_plot, uniform_plot),
            ReplacementTransform(top_label, uniform_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This transformation is the heart of the Central Limit Theorem.
        self.lecture[4].set_color(COLOR_LINE)
        
        # Mean line through the center
        center_line = Line(
            bottom_axes.c2p(0, 0),
            bottom_axes.c2p(0, 1.1),
            color=COLOR_LINE,
            stroke_width=4
        )
        # Fix: Replace MathTex with Text to avoid LaTeX requirement dependency
        mean_label = Text("μ", color=COLOR_LINE, font_size=36)
        # Fix from VideoCritic: Move to F5, scale 0.6
        self.place_at_grid(mean_label, 'F5', scale_factor=0.6)
        
        clt_text = Text("Central Limit Theorem", font_size=24, color=COLOR_CLT)
        self.place_at_grid(clt_text, 'E5', scale_factor=0.8) # Adjusted for visibility

        self.play(
            Create(center_line), 
            Write(mean_label),
            FadeIn(clt_text),
            Flash(clt_text, color=COLOR_CLT)
        )
        self.wait(2)
