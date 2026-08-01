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
        # Setup layout
        lecture_lines = [
            "Formulas provide shortcuts for finding local steepness.",
            "The speedometer shows how fast the curve changes.",
            "You now have the intuition to master calculus!"
        ]
        self.setup_layout("Summary: The Speedometer of Change", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Simultaneously draw a white curve on the left and a speedometer asset on the right.
        self.lecture[0].set_color(WHITE)
        
        # 1. Curve Setup
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[0, 4, 1],
            x_length=3,
            y_length=3,
            axis_config={"color": GREY_B, "include_tip": False}
        )
        self.place_in_area(axes, "A1", "D3") # Resolved Issue 40
        
        curve = axes.plot(lambda x: x**2, x_range=[-1.5, 1.5], color=WHITE)
        curve_label = Text("y = x^2", font_size=18, color=WHITE)
        curve_label.next_to(curve, UP, buff=0.1)

        # 2. Speedometer Setup (Resolved Issue 29)
        speedo_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/speed.svg")
        speedo_asset.set_color(WHITE)
        speedo_label = Text("Slope Gauge", font_size=18, color=WHITE)
        
        speedo_group = VGroup(speedo_asset, speedo_label)
        self.place_in_area(speedo_group, "A4", "D6", scale_factor=0.8) # Resolved Issue 41
        speedo_label.next_to(speedo_asset, DOWN, buff=0.2)
        
        speedo_center = speedo_asset.get_center()

        # 3. Needle setup
        needle_len = speedo_asset.height * 0.45
        needle = Line(speedo_center, speedo_center + UP * needle_len, color="#FF4500", stroke_width=5)
        needle_pivot = Dot(speedo_center, color=WHITE, radius=0.05)

        self.play(
            Create(axes),
            Create(curve),
            Write(curve_label),
            FadeIn(speedo_asset),
            Write(speedo_label),
            Create(needle),
            Create(needle_pivot),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The speedometer needle (#FF4500) rotates based on the curve's current slope.
        self.lecture[1].set_color("#FF4500")

        # Value tracker for x position on curve
        x_tracker = ValueTracker(-1.5)
        
        # Moving dot on curve
        moving_dot = Dot(color="#FF4500").add_updater(
            lambda d: d.move_to(axes.c2p(x_tracker.get_value(), x_tracker.get_value()**2))
        )
        
        # Tangent line (visual shortcut for slope)
        tangent = Line(color="#FF4500", stroke_width=4)
        def update_tangent(t):
            val = x_tracker.get_value()
            alpha = (val + 1.5) / 3
            point = curve.point_from_proportion(alpha)
            # Derivative of x^2 is 2x. Slope = 2*val.
            # We construct a line segment through 'point' with slope mapping to axes.
            # Faster way: use small offset to find direction
            eps = 0.001
            p1 = curve.point_from_proportion(max(0, alpha - eps))
            p2 = curve.point_from_proportion(min(1, alpha + eps))
            tangent_vec = normalize(p2 - p1)
            t.set_points_as_corners([point - tangent_vec * 0.7, point + tangent_vec * 0.7])
        
        tangent.add_updater(update_tangent)

        # Needle rotation logic
        def update_needle(n):
            current_x = x_tracker.get_value()
            slope = 2 * current_x
            # Map slope (-inf, inf) to angles (PI, 0)
            angle = PI/2 - np.arctan(slope)
            direction = np.array([np.cos(angle), np.sin(angle), 0])
            n.set_points_as_corners([speedo_center, speedo_center + direction * needle_len])

        needle.add_updater(update_needle)
        
        self.add(moving_dot, tangent)
        
        # Animate movement
        self.play(x_tracker.animate.set_value(1.5), run_time=5, rate_func=linear)
        self.play(x_tracker.animate.set_value(-1.5), run_time=3, rate_func=linear)
        self.play(x_tracker.animate.set_value(0), run_time=2, rate_func=smooth)
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Flash the text 'Derivative = Speed of Change' in white across the center.
        self.lecture[2].set_color(WHITE)
        
        flash_text = Text("Derivative = Speed of Change", font_size=28, color=WHITE, weight=BOLD)
        self.place_in_area(flash_text, "E1", "F6", scale_factor=0.8) # Resolved Issue 39
        
        # Add a dark semi-transparent rectangle behind the flash text for readability
        bg_rect = SurroundingRectangle(flash_text, color=BLACK, fill_opacity=0.8, buff=0.2)

        self.play(
            FadeIn(bg_rect),
            Write(flash_text),
            flash_text.animate.scale(1.1),
            run_time=0.8
        )
        self.play(flash_text.animate.scale(1/1.1), run_time=0.8)
        self.wait(2)

        # Final cleanup/freeze
        needle.remove_updater(update_needle)
        tangent.remove_updater(update_tangent)
        moving_dot.remove_updater(None)
        self.wait(1)
