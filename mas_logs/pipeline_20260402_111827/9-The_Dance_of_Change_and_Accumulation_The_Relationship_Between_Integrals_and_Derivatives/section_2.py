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
        # Setup layout with the official lines
        title = "The Derivative: Zooming into the Slope"
        lines = [
            "A curve represents varying movement over time.",
            "Let's focus on one specific point.",
            "Zooming in reveals a straight line segment.",
            "This tangent's slope is the derivative.",
            "The derivative changes as the slope fluctuates."
        ]
        self.setup_layout(title, lines)

        # Helper colors
        CURVE_COLOR = "#FFFF00"    # Yellow
        DOT_COLOR = "#FF0000"      # Red
        ZOOM_COLOR = "#FFFFFF"     # White
        TANGENT_COLOR = "#00BFFF"  # Deep Sky Blue
        CHANGE_COLOR = "#00FF00"   # Green

        # === Animation for Lecture Line 1 ===
        # A curved function f(x) = sin(x) + 2 is drawn in yellow
        self.lecture[0].set_color(CURVE_COLOR)
        
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 4, 1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": True, "color": GREY_B}
        )
        self.place_in_area(axes, "B2", "F6")
        
        curve = axes.plot(lambda x: np.sin(x) + 2, x_range=[0, 5], color=CURVE_COLOR)
        
        self.play(Create(axes), Create(curve))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A red dot appears on the curve, and a magnifying circle focus on it.
        self.lecture[1].set_color(DOT_COLOR)
        
        # ValueTracker for the dot's x position
        x_tracker = ValueTracker(1.5)
        
        dot = Dot(color=DOT_COLOR)
        dot.add_updater(lambda d: d.move_to(axes.c2p(x_tracker.get_value(), np.sin(x_tracker.get_value()) + 2)))
        
        magnifier = Circle(radius=0.7, color=ZOOM_COLOR, stroke_width=3)
        magnifier.add_updater(lambda m: m.move_to(dot.get_center()))
        
        self.play(FadeIn(dot), Create(magnifier))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Inside the circle, the curve is zoomed in until it appears as a straight line segment.
        self.lecture[2].set_color(ZOOM_COLOR)
        
        # Masking the left side to prevent axes from overlapping lecture text during zoom
        mask = Rectangle(width=6, height=10, color=BLACK, fill_opacity=1).to_edge(LEFT, buff=0)
        mask.set_z_index(10)
        self.add(mask)
        self.lecture.set_z_index(11)
        self.title.set_z_index(11)
        
        # Zoom effect: scale the entire coordinate system around the dot
        zoom_factor = 6
        initial_dot_pos = dot.get_center().copy()
        
        # Save state for Line 5 fluctuation
        axes.save_state()
        curve.save_state()
        
        self.play(
            axes.animate.scale(zoom_factor, about_point=initial_dot_pos),
            curve.animate.scale(zoom_factor, about_point=initial_dot_pos),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # A tangent line is drawn through the dot, with a label 'Slope = Derivative'.
        self.lecture[3].set_color(TANGENT_COLOR)
        
        def get_tangent_line(mob):
            x = x_tracker.get_value()
            p_center = axes.c2p(x, np.sin(x) + 2)
            # Use small delta to find direction on scaled axes
            dx = 0.005
            p1 = axes.c2p(x - dx, np.sin(x - dx) + 2)
            p2 = axes.c2p(x + dx, np.sin(x + dx) + 2)
            dir_vec = normalize(p2 - p1)
            # Fixed visual length
            length = 2.5
            mob.set_points_as_corners([p_center - dir_vec * length/2, p_center + dir_vec * length/2])

        tangent_line = Line(ORIGIN, RIGHT, color=TANGENT_COLOR, stroke_width=4)
        tangent_line.add_updater(get_tangent_line)
        
        slope_label = Text("Slope = Derivative", font_size=24, color=TANGENT_COLOR)
        # Fix for Issue 30 and 31: centering label over the animation area
        self.place_in_area(slope_label, "A4", "A6", scale_factor=0.8)
        
        self.play(Create(tangent_line), FadeIn(slope_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The dot moves along the curve, and the tangent line rotates to match the changing slope dynamically.
        self.lecture[4].set_color(CHANGE_COLOR)
        
        # Zoom back out slightly to see the curve's fluctuation better
        self.play(
            axes.animate.restore(),
            curve.animate.restore(),
            run_time=1.5
        )
        
        # Animate movement
        self.play(
            x_tracker.animate.set_value(3.5),
            run_time=4,
            rate_func=linear
        )
        self.wait(1)
        
        # Return movement
        self.play(
            x_tracker.animate.set_value(1.0),
            run_time=3,
            rate_func=smooth
        )
        self.wait(2)
