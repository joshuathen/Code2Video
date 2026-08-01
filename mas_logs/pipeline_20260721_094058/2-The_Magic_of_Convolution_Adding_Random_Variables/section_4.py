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
        # Setup the layout
        self.setup_layout("Transition to Continuous: The Convolution Integral", [
            "- For continuous variables, we use the convolution integral.",
            "- It calculates the probability density of the sum.",
            "- Visualize two smooth Bell curves passing through each other.",
            "- The area of their overlap determines the sum's PDF.",
            "- This integral formalizes the 'Flip and Slide' intuition."
        ])

        # Color definitions
        COLOR_INT = "#FFFFFF"
        COLOR_GAUSS1 = "#ADD8E6"  # Light Blue
        COLOR_GAUSS2 = "#FFB6C1"  # Light Pink
        COLOR_OVERLAP = "#00FFFF" # Cyan

        # === Animation for Lecture Line 1 ===
        # "For continuous variables, we use the convolution integral."
        self.lecture[0].set_color(COLOR_INT)
        sigma = MathTex(r"\sum", color=COLOR_INT)
        self.place_at_grid(sigma, "B3", scale_factor=1.0)
        
        self.play(Write(sigma))
        self.wait(1)
        
        integral_sign = MathTex(r"\int", color=COLOR_INT)
        self.place_at_grid(integral_sign, "B3", scale_factor=1.0)
        
        self.play(Transform(sigma, integral_sign))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "It calculates the probability density of the sum."
        self.lecture[1].set_color(COLOR_INT)
        # formula replaces the transformed integral_sign (sigma)
        formula = MathTex(
            r"f_Z(z) = \int_{-\infty}^{\infty} f_X(x) f_Y(z-x) dx",
            color=COLOR_INT
        )
        self.place_in_area(formula, "B2", "B6", scale_factor=0.8)
        
        self.play(Transform(sigma, formula))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # "Visualize two smooth Bell curves passing through each other."
        self.lecture[2].set_color(COLOR_GAUSS1)
        
        axes = Axes(
            x_range=[-3.5, 3.5, 1],
            y_range=[0, 1, 0.5],
            axis_config={"include_tip": False},
            x_length=4.5,
            y_length=2.5
        )
        self.place_in_area(axes, "D2", "F6", scale_factor=1.0)
        
        def gaussian_func(x, mu, sigma_val=0.5):
            return np.exp(-((x - mu) ** 2) / (2 * sigma_val ** 2)) / (sigma_val * np.sqrt(2 * np.pi))

        # Static Gaussian 1 (Light Blue)
        g1 = axes.plot(lambda x: gaussian_func(x, 0), color=COLOR_GAUSS1)
        
        # Moving Gaussian 2 (Light Pink)
        z_tracker = ValueTracker(-2.5)
        
        # VMobject for g2 to allow efficient point updates via add_updater
        g2 = VMobject(color=COLOR_GAUSS2)
        def update_g2(m):
            z = z_tracker.get_value()
            x_vals = np.linspace(-3.5, 3.5, 80)
            points = [axes.c2p(x, gaussian_func(x, z)) for x in x_vals]
            m.set_points_as_corners(points)
        
        g2.add_updater(update_g2)
        update_g2(g2) # Initial state
        
        self.play(Create(axes), Create(g1), FadeIn(g2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "The area of their overlap determines the sum's PDF."
        self.lecture[3].set_color(COLOR_OVERLAP)

        # Overlap Area VMobject with Cyan color
        overlap_fill = VMobject(color=COLOR_OVERLAP, fill_opacity=0.5, stroke_width=0)
        def update_overlap(m):
            z = z_tracker.get_value()
            x_vals = np.linspace(-3.5, 3.5, 80)
            points = []
            # Start at left edge on x-axis
            points.append(axes.c2p(-3.5, 0))
            for x in x_vals:
                # Value is the overlap (minimum of the two densities)
                val = min(gaussian_func(x, 0), gaussian_func(x, z))
                points.append(axes.c2p(x, val))
            # Close at right edge on x-axis
            points.append(axes.c2p(3.5, 0))
            m.set_points_as_corners(points)
        
        overlap_fill.add_updater(update_overlap)
        update_overlap(overlap_fill)
        
        self.play(FadeIn(overlap_fill))
        # Slide through and show overlap changing
        self.play(z_tracker.animate.set_value(0), run_time=2.5, rate_func=linear)
        self.play(z_tracker.animate.set_value(2.5), run_time=2.5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "This integral formalizes the 'Flip and Slide' intuition."
        self.lecture[4].set_color(COLOR_INT)
        
        # Smooth swing back to visualize the sliding movement again
        self.play(z_tracker.animate.set_value(-2.5), run_time=2, rate_func=smooth)
        self.play(z_tracker.animate.set_value(0), run_time=2, rate_func=smooth)
        self.wait(2)
