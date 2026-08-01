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
        lecture_lines = [
            "To calculate the sum, we use the convolution operation.",
            "First, we flip one of the probability distributions horizontally.",
            "Then, we slide this flipped distribution across the other one.",
            "At each step, we multiply the overlapping values together.",
            "The integral of this product gives the new probability density."
        ]
        self.setup_layout("The Mechanics: Flip, Slide, and Multiply", lecture_lines)

        # Colors
        color_fx = "#87CEEB"
        color_fy = "#DA70D6"
        color_overlap = "#FFFFE0"
        color_fz = "#32CD32"

        # Axes setup - Fixing Issue 28 (B2-D6 area)
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 1.2, 0.5],
            x_length=5,
            y_length=3,
            axis_config={"include_tip": False}
        ).scale(0.8)
        self.place_in_area(axes, "B2", "D6")

        # Result Axes (smaller, below)
        res_axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.6, 0.2],
            x_length=5,
            y_length=1.5,
            axis_config={"include_tip": False}
        ).scale(0.8)
        self.place_in_area(res_axes, "E2", "F6")
        
        # Labels - Fixing Issue 29 (label_fx at B2) and Issue 30 (label_fz at E2)
        label_fx = Text("f_X(x)", color=color_fx, font_size=24)
        label_fy = Text("f_Y(x)", color=color_fy, font_size=24)
        label_fz = Text("f_Z(z)", color=color_fz, font_size=24)
        
        self.place_at_grid(label_fx, "B2", scale_factor=0.8)
        self.place_at_grid(label_fy, "B6", scale_factor=0.8)
        self.place_at_grid(label_fz, "E2", scale_factor=0.8)

        # Probability Density Functions
        def fx_func(x):
            return np.exp(-x**2)
        
        def fy_func(x):
            return 0.8 * np.exp(-(x-0.5)**2)

        curve_fx = axes.plot(fx_func, color=color_fx)
        curve_fy = axes.plot(fy_func, color=color_fy)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(axes), Create(res_axes), Write(label_fx), Write(label_fy), Write(label_fz))
        self.play(Create(curve_fx), Create(curve_fy))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        def fy_flipped_func(x):
            # Mirror the original around its local center or just x=0? 
            # Storyboard says "Flip horizontally".
            return 0.8 * np.exp(-(-x-0.5)**2)
        
        curve_fy_flipped = axes.plot(fy_flipped_func, color=color_fy)
        label_fy_flipped = Text("f_Y(-x)", color=color_fy, font_size=24)
        self.place_at_grid(label_fy_flipped, "B6", scale_factor=0.8)

        self.play(
            ReplacementTransform(curve_fy, curve_fy_flipped),
            ReplacementTransform(label_fy, label_fy_flipped)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        z_tracker = ValueTracker(-4)
        
        # Sliding curve: f_Y(z-x)
        # We use always_redraw here for the dynamic plotting against the fixed axes.
        curve_fy_sliding = always_redraw(lambda: 
            axes.plot(lambda x: 0.8 * np.exp(-(z_tracker.get_value()-x-0.5)**2), color=color_fy)
        )
        
        label_fy_sliding = Text("f_Y(z-x)", color=color_fy, font_size=24)
        self.place_at_grid(label_fy_sliding, "B6", scale_factor=0.8)

        self.play(
            ReplacementTransform(curve_fy_flipped, curve_fy_sliding),
            ReplacementTransform(label_fy_flipped, label_fy_sliding)
        )
        
        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Product area (overlap) - f_X(x) * f_Y(z-x)
        overlap_area = always_redraw(lambda: 
            axes.get_area(
                axes.plot(lambda x: fx_func(x) * (0.8 * np.exp(-(z_tracker.get_value()-x-0.5)**2)), color=color_overlap),
                x_range=[-4, 4],
                color=color_overlap,
                opacity=0.5
            )
        )
        self.add(overlap_area)
        
        # The true convolution of two Gaussians is a Gaussian.
        # sigma1=1/sqrt(2), sigma2=1/sqrt(2). sigma_new = 1.
        # Here mu1=0, mu2=0.5, so mu_new=0.5.
        def fz_val(z):
            # Approximate height for visual effect
            return 0.5 * np.exp(-(z-0.5)**2 / 2)

        # Dot showing the current value on result axes
        dot_fz = always_redraw(lambda: 
            Dot(res_axes.c2p(z_tracker.get_value(), fz_val(z_tracker.get_value())), color=color_fz, radius=0.05)
        )
        # Trace the path of the dot
        tracing_curve = TracedPath(dot_fz.get_center, stroke_color=color_fz, stroke_width=3)
        
        self.add(dot_fz, tracing_curve)
        
        # Slow slide to see the area and the tracing
        self.play(z_tracker.animate.set_value(4), run_time=6, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Finish the curve
        fz_curve_final = res_axes.plot(fz_val, color=color_fz, x_range=[-4, 4])
        self.play(Create(fz_curve_final))
        self.wait(2)
