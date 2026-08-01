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
        # Fetching titles and lines from storyboard/outline
        title = "The Core Theorem (The Magic Rule)"
        lines = [
            "This is the Central Limit Theorem in action.",
            "The center stays at the true population mean.",
            "Increasing sample size makes the curve much narrower.",
            "With enough samples, the shape is always Normal.",
            "Chaos transforms into predictable, mathematical order."
        ]
        self.setup_layout(title, lines)

        # Colors from storyboard
        GREEN = "#00FF00"
        RED = "#FF0000"
        WHITE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Display the formula X-bar ~ N(mu, sigma squared over n)
        # Resolved Issue 35: formula at A3-A6, scale 0.8
        formula = MathTex(r"\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right)", color=WHITE_COLOR)
        self.place_in_area(formula, 'A3', 'A6', scale_factor=0.8)
        
        self.lecture[0].set_color(WHITE_COLOR)
        self.play(Write(formula), run_time=1.5)
        self.wait(1)

        # Pre-creating elements for the plot group to handle dynamic updates
        n_tracker = ValueTracker(1)
        
        # Initial points for curve at n=1
        def get_curve_points(n_val):
            sigma_base = 1.0
            s = sigma_base / np.sqrt(n_val)
            h = 0.5 * np.sqrt(n_val)
            x_vals = np.linspace(-2.5, 2.5, 100)
            return [np.array([x, h * np.exp(-0.5 * (x / s)**2), 0]) for x in x_vals]

        # Base curve mobject
        curve = VMobject(color=GREEN, stroke_width=4)
        curve.set_points_as_corners(get_curve_points(1))
        curve.make_smooth()
        
        # Constant center line mu
        mean_line = Line(ORIGIN, UP * 3, color=RED, stroke_width=3)
        mu_label = MathTex(r"\mu", color=RED).scale(0.8)
        mu_label.next_to(mean_line, UP, buff=0.1)
        
        # Group them for placement as per Issue 34
        plot_group = VGroup(curve, mean_line, mu_label)
        
        # Resolved Issue 34: Anchoring the plot group to C2-F6
        self.place_in_area(plot_group, 'C2', 'F6', scale_factor=0.7)
        
        # Scale factor for internal point updates
        group_scale = 0.7

        # Updater to adjust curve points relative to the anchored baseline
        def update_curve(mob):
            n_val = n_tracker.get_value()
            pts = get_curve_points(n_val)
            # Transform local coordinates to scene coordinates
            anchor = mean_line.get_start() # Baseline of the mean line
            scene_pts = [anchor + group_scale * p for p in pts]
            mob.set_points_as_corners(scene_pts)
            mob.make_smooth()

        curve.add_updater(update_curve)

        # === Animation for Lecture Line 2 ===
        # Highlight the constant center line mu in red
        self.lecture[1].set_color(RED)
        self.play(Create(mean_line), Write(mu_label), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show wide curve and increase n
        # Resolved Issue 36: n_group at B6, scale 0.8
        n_text_label = Text("n = ", font_size=24, color=WHITE)
        n_value = DecimalNumber(1, num_decimal_places=0, font_size=24, color=WHITE)
        n_group = VGroup(n_text_label, n_value).arrange(RIGHT, buff=0.1)
        self.place_at_grid(n_group, 'B6', scale_factor=0.8)
        n_value.add_updater(lambda d: d.set_value(n_tracker.get_value()))
        
        self.lecture[2].set_color(GREEN)
        self.play(Create(curve), Write(n_group), run_time=1)
        # Squeeze the green bell curve to make it taller and narrower
        self.play(n_tracker.animate.set_value(10), run_time=2, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Final narrowing to n=30, indicating Normal shape
        self.lecture[3].set_color(GREEN)
        self.play(n_tracker.animate.set_value(30), run_time=2, rate_func=linear)
        self.play(Indicate(curve, color=GREEN, scale_factor=1.1))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Result highlight: chaos to mathematical order
        self.lecture[4].set_color(WHITE_COLOR)
        box = SurroundingRectangle(formula, color=WHITE_COLOR, buff=0.15)
        self.play(Create(box))
        self.wait(2)

        # Final cleanup: reset lecture colors
        for line in self.lecture:
            line.set_color(WHITE)
        self.wait(1)
