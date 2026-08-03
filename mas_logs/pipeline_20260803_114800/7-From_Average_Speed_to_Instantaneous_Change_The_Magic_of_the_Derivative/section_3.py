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
        # Data from storyboard
        title_text = "The Secant Line Animation"
        lecture_lines = [
            "Most real-world movement follows a curved path.",
            "A secant line connects two points on the curve.",
            "Its slope shows the average speed between them."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors as specified
        curve_color = "#FFFFFF"   # White
        point_color = "#ADD8E6"   # Light Blue
        secant_color = "#0000FF"  # Blue

        # === Animation for Lecture Line 1 ===
        # Highlight matching the curve color
        self.lecture[0].set_color(curve_color)

        # Setup Axes
        axes = Axes(
            x_range=[0, 3, 1],
            y_range=[0, 9, 2],
            axis_config={"include_tip": True, "color": WHITE},
            x_length=4,
            y_length=4
        )
        self.place_in_area(axes, "A1", "F6")
        
        # Create curve y = x^2
        curve = axes.plot(lambda x: x**2, x_range=[0, 2.5], color=curve_color)
        curve_label = MathTex("y = x^2", font_size=24, color=curve_color)
        # Resolved Issue 25: move curve_label to B5
        self.place_at_grid(curve_label, "B5", scale_factor=0.8)

        self.play(Create(axes), Create(curve), Write(curve_label), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Update colors: dim previous, highlight current with point/line color
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color(point_color)

        # Resolved Issue 20: Use vehicle asset for point B
        vehicle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vehicle.svg")
        vehicle.set_color(point_color)
        vehicle.scale(0.2)

        # Points A and B with ValueTracker for B
        x_a = 0.5
        x_b_start = 2.0
        x_b_tracker = ValueTracker(x_b_start)
        
        dot_a = Dot(axes.c2p(x_a, x_a**2), color=point_color)
        label_a = MathTex("A", font_size=24, color=point_color)
        label_a.next_to(dot_a, LEFT, buff=0.1)
        
        # Vehicle represents point B
        vehicle.move_to(axes.c2p(x_b_start, x_b_start**2))
        vehicle.add_updater(lambda v: v.move_to(axes.c2p(x_b_tracker.get_value(), x_b_tracker.get_value()**2)))
        
        label_b = MathTex("B", font_size=24, color=point_color)
        label_b.add_updater(lambda l: l.next_to(vehicle, RIGHT, buff=0.1))
        
        # Secant Line connecting A and B
        secant_line = Line(axes.c2p(x_a, x_a**2), axes.c2p(x_b_start, x_b_start**2), color=secant_color, stroke_width=4)
        
        def update_secant(line):
            start = axes.c2p(x_a, x_a**2)
            end = axes.c2p(x_b_tracker.get_value(), x_b_tracker.get_value()**2)
            line.set_points_by_ends(start, end)
        
        secant_line.add_updater(update_secant)

        self.play(
            FadeIn(dot_a), Write(label_a),
            FadeIn(vehicle), Write(label_b),
            Create(secant_line),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Update colors: dim previous, highlight current with secant color
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color(secant_color)

        # Label for the slope
        slope_label = Text("Slope = Average Speed", font_size=20, color=secant_color)
        # Resolved Issue 24: placement to F2-F4 area
        self.place_in_area(slope_label, "F2", "F4", scale_factor=0.6)
        
        self.play(Write(slope_label))
        self.wait(1)

        # Animate point B (vehicle) sliding along the curve toward point A
        self.play(
            x_b_tracker.animate.set_value(0.7),
            run_time=4,
            rate_func=linear
        )
        self.wait(2)

        # Final cleanup: reset colors
        self.play(self.lecture.animate.set_color(WHITE), run_time=1)
        self.wait(2)
