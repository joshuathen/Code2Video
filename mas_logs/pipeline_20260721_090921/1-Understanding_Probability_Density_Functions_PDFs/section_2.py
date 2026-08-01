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
        # --- Data Setup ---
        title = "Prerequisite: The Vanishing Histogram"
        lines = [
            "Histograms show data frequency using rectangular bins.",
            "Narrower bins make the steps look smoother.",
            "As bin width approaches zero, the steps vanish.",
            "The jagged histogram becomes a smooth curve.",
            "This curve is the Probability Density Function."
        ]
        self.setup_layout(title, lines)

        # Colors
        ORANGE_COLOR = "#FFA500"
        YELLOW_COLOR = "#FFFF00"
        GREY_COLOR = "#888888"

        # --- Axes Construction ---
        # The axes represent the probability space
        # Using a range that accommodates a standard normal distribution
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 0.5, 0.1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": False, "color": GREY_COLOR},
            tips=False
        )
        
        # Position axes in the grid area as suggested by VideoCritic
        # Fixing issues 22, 23, 24
        self.place_in_area(axes, "B2", "F5", scale_factor=0.75)

        # Probability Density Function: Normal Distribution
        def pdf_func(x):
            return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

        # Helper to create histogram bins
        def get_histogram_bins(num_bins, color):
            bin_width = 6.0 / num_bins
            bins = VGroup()
            for i in range(num_bins):
                x_start = -3 + i * bin_width
                x_mid = x_start + bin_width / 2
                height = pdf_func(x_mid)
                
                # Calculate physical dimensions based on coordinate system
                p_bottom = axes.c2p(x_start, 0)
                p_top = axes.c2p(x_start + bin_width, height)
                
                rect_w = p_top[0] - p_bottom[0]
                rect_h = p_top[1] - p_bottom[1]
                
                rect = Rectangle(
                    width=rect_w,
                    height=rect_h,
                    fill_color=color,
                    fill_opacity=0.6,
                    stroke_color=color,
                    stroke_width=0.5
                )
                # Position the rectangle center properly
                rect.move_to(axes.c2p(x_mid, height/2))
                bins.add(rect)
            return bins

        # === Animation for Lecture Line 1 ===
        # Histograms show data frequency using rectangular bins.
        self.play(self.lecture[0].animate.set_color(ORANGE_COLOR))
        hist_wide = get_histogram_bins(5, ORANGE_COLOR)
        self.play(Create(axes), FadeIn(hist_wide))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Narrower bins make the steps look smoother.
        self.play(self.lecture[1].animate.set_color(ORANGE_COLOR))
        hist_medium = get_histogram_bins(10, ORANGE_COLOR)
        self.play(ReplacementTransform(hist_wide, hist_medium))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # As bin width approaches zero, the steps vanish.
        self.play(self.lecture[2].animate.set_color(ORANGE_COLOR))
        hist_thin = get_histogram_bins(20, ORANGE_COLOR)
        self.play(ReplacementTransform(hist_medium, hist_thin))
        
        hist_very_thin = get_histogram_bins(50, ORANGE_COLOR)
        self.play(ReplacementTransform(hist_thin, hist_very_thin), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The jagged histogram becomes a smooth curve.
        self.play(self.lecture[3].animate.set_color(YELLOW_COLOR))
        pdf_curve = axes.plot(pdf_func, color=YELLOW_COLOR, stroke_width=4)
        self.play(Create(pdf_curve))
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        # This curve is the Probability Density Function.
        self.play(self.lecture[4].animate.set_color(YELLOW_COLOR))
        self.play(FadeOut(hist_very_thin))
        self.play(Indicate(pdf_curve, color=YELLOW_COLOR))
        self.wait(3)
