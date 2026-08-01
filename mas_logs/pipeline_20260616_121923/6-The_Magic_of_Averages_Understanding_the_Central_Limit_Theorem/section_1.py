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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Prerequisite: The Individual vs. The Average", 
            [
                'Every population has a mean, labeled mu.', 
                'Standard deviation, sigma, measures the spread.', 
                'Individual data points are often messy and unpredictable.', 
                "Let's look at heights in a giraffe forest.", 
                'This raw distribution is chaotic and skewed.'
            ]
        )

        # --- Object Creation ---

        # 1. Distribution Chart Elements
        axis = Line(LEFT * 2.5, RIGHT * 2.5, color=WHITE)
        height_label = Text("Height", color=WHITE)
        
        # Mean line shifted slightly left to reflect a skewed distribution center
        mean_line = Line(DOWN * 0.2, UP * 2.5, color=WHITE, stroke_width=4).shift(LEFT * 1.0)
        mu_label = Text("μ", color=WHITE)

        # 100 Dots (#ADD8E6) representing a messy population
        np.random.seed(42)
        dots = VGroup()
        for _ in range(100):
            # Log-normal distribution for a chaotic, skewed forest look
            x_val = np.clip(np.random.lognormal(mean=0, sigma=0.5) - 1.8, -2.4, 2.4)
            y_val = np.random.uniform(0.1, 2.5)
            dot = Dot(point=[x_val, y_val, 0], radius=0.05, color="#ADD8E6", fill_opacity=0.7)
            dots.add(dot)

        # Skewed Curve (#FFA500)
        curve_points = [
            [-2.4, 0.1, 0], [-1.5, 2.8, 0], [-0.5, 1.8, 0], 
            [0.5, 0.9, 0], [1.5, 0.4, 0], [2.4, 0.1, 0]
        ]
        skewed_curve = VMobject(color="#FFA500").set_points_as_corners(curve_points).make_smooth()

        # Master group for grid anchoring (Issue 39)
        distribution_group = VGroup(axis, dots, skewed_curve, mean_line)
        
        # 2. Sigma Indicator
        sigma_arrow = DoubleArrow(start=[-1.0, 1.8, 0], end=[0.2, 1.8, 0], 
                                  color="#ADD8E6", tip_length=0.1, buff=0)
        sigma_label = Text("σ", color="#ADD8E6")

        # --- Positioning (Resolving Issues) ---

        # Issue 39: Anchor distribution chart to the specified area
        self.place_in_area(distribution_group, 'B2', 'F6', scale_factor=0.8)
        
        # Issue 40: Position mean label at B3
        self.place_at_grid(mu_label, 'B3', scale_factor=1.0)
        
        # Issue 41: Position height label at F5
        self.place_at_grid(height_label, 'F5', scale_factor=0.7)
        
        # Position Sigma indicator at C4
        self.place_at_grid(sigma_arrow, 'C4', scale_factor=0.8)
        sigma_label.scale(0.8).next_to(sigma_arrow, UP, buff=0.1)

        # --- Animation Sequence ---

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(Create(axis), Write(height_label))
        self.play(Create(mean_line), Write(mu_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#ADD8E6"))
        self.play(Create(sigma_arrow), Write(sigma_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#ADD8E6"))
        # Fade in the chaotic dots
        self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.01), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFD700"))
        # Highlight individual circles by changing their color to #FFD700
        highlighted_indices = [7, 15, 32, 48, 88]
        self.play(*[dots[i].animate.set_color("#FFD700").scale(1.5) for i in highlighted_indices])
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFA500"))
        # Draw the highly skewed curve
        self.play(Create(skewed_curve))
        self.wait(2)
