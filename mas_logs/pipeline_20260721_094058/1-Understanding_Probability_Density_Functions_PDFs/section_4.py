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
        # Data from shared state
        title_text = "The Two Golden Rules"
        lecture_lines = [
            "Probability density can never be negative.",
            "The curve must stay above the horizontal axis.",
            "Total area under the curve must equal one."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Color constants
        PURPLE = "#944243"
        WHITE_TINT = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Goal: Show a curve where a part below the x-axis flips to purple #944243 above it.
        self.play(self.lecture[0].animate.set_color(PURPLE))

        # Setup Axes on the right half
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-1, 2, 1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": False, "color": WHITE}
        )
        self.place_in_area(axes, "B1", "F6", scale_factor=0.8)
        
        # Initial curve with negative part: f(x) = 0.5 * (x^2 - 1)
        def initial_func(x):
            return 0.5 * (x**2 - 1)
        
        curve = axes.plot(initial_func, color=WHITE)
        
        self.play(Create(axes), Create(curve))
        self.wait(0.5)

        # Flip part below x-axis to be positive
        def positive_func(x):
            return abs(0.5 * (x**2 - 1))
        
        curve_flipped = axes.plot(positive_func, color=PURPLE)
        
        self.play(Transform(curve, curve_flipped))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Goal: Highlight that the curve stays above the horizontal axis.
        self.play(self.lecture[1].animate.set_color(PURPLE))
        
        # Highlight x-axis to emphasize boundary
        self.play(Indicate(axes.x_axis, color=PURPLE))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Goal: Fill the entire area under the curve with white tint.
        # Goal: Display 'Total Area = 1' in the center.
        self.play(self.lecture[2].animate.set_color(WHITE_TINT))

        # Transition to a standard PDF shape (Gaussian-like)
        def bell_curve_func(x):
            return 1.5 * np.exp(-(x**2))
        
        bell_curve = axes.plot(bell_curve_func, color=WHITE_TINT)
        area_fill = axes.get_area(bell_curve, x_range=[-2, 2], color=WHITE_TINT, opacity=0.3)
        
        # Label text - Fixed as per Issue 27, 28, 29
        area_label = Text("Total Area = 1", font_size=24, color=WHITE_TINT)
        self.place_in_area(area_label, "A3", "B5", scale_factor=0.8)

        self.play(
            FadeOut(curve),
            Create(bell_curve),
            FadeIn(area_fill),
            Write(area_label)
        )
        self.wait(2)
