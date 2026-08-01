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
        # Data and setup
        title_text = "Graphical Synthesis: Slopes vs. Areas"
        lines = [
            "Look at position and velocity graphs side-by-side.",
            "The derivative's height is the position's slope.",
            "The integral's area is the position's vertical change.",
            "Watch as slope and area change in perfect sync.",
            "This bridge connects rates of change to total accumulation."
        ]
        
        self.setup_layout(title_text, lines)
        
        # Colors
        color_pos = "#FF69B4"  # Hot Pink
        color_vel = "#00CED1"  # Dark Turquoise (Cyan)
        highlight_color = YELLOW
        
        # === Animation for Lecture Line 1 ===
        # "Look at position and velocity graphs side-by-side."
        self.lecture[0].set_color(highlight_color)
        
        # Setup Left Graph (Position)
        # Fix Issue 38: Move axes_left to 'B1'-'F3'
        axes_left = Axes(
            x_range=[0, 2.2, 1],
            y_range=[0, 4.5, 1],
            x_length=2.5,
            y_length=4.0,
            axis_config={"include_tip": True, "color": GREY_A}
        )
        self.place_in_area(axes_left, 'B1', 'F3', scale_factor=0.8)
        
        # Fix Issue 40: Scale label_pos to 0.8
        label_pos = Text("Position s(t)", font_size=18, color=color_pos)
        self.place_at_grid(label_pos, 'A2', scale_factor=0.8)
        
        # Setup Right Graph (Velocity)
        # Fix Issue 39: Move axes_right to 'B4'-'F6'
        axes_right = Axes(
            x_range=[0, 2.2, 1],
            y_range=[0, 4.5, 1],
            x_length=2.5,
            y_length=4.0,
            axis_config={"include_tip": True, "color": GREY_A}
        )
        self.place_in_area(axes_right, 'B4', 'F6', scale_factor=0.8)
        
        # Fix Issue 40: Scale label_vel to 0.8
        label_vel = Text("Velocity v(t)", font_size=18, color=color_vel)
        self.place_at_grid(label_vel, 'A5', scale_factor=0.8)
        
        # Load Asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg]
        bridge_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg", color=WHITE)
        self.place_in_area(bridge_icon, 'A3', 'A4', scale_factor=0.3)
        
        # Curves
        curve_pos = axes_left.plot(lambda t: t**2, x_range=[0, 2], color=color_pos)
        curve_vel = axes_right.plot(lambda t: 2*t, x_range=[0, 2], color=color_vel)
        
        self.play(
            Create(axes_left), Create(axes_right),
            Write(label_pos), Write(label_vel),
            FadeIn(bridge_icon),
            run_time=1
        )
        self.play(Create(curve_pos), Create(curve_vel), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "The derivative's height is the position's slope."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(highlight_color)
        
        t_tracker = ValueTracker(1.2)
        
        # Slope elements (Left side)
        dot_pos = Dot(color=color_pos)
        dot_pos.add_updater(lambda d: d.move_to(axes_left.c2p(t_tracker.get_value(), t_tracker.get_value()**2)))
        
        # Tangent line segment on the parabola
        tangent = Line(color=WHITE, stroke_width=4)
        def update_tangent(mob):
            t = t_tracker.get_value()
            if t < 0.1: t = 0.1
            slope = 2*t
            # Tangent line segment centered at t
            p1_x = t - 0.25
            p2_x = t + 0.25
            p1 = axes_left.c2p(p1_x, t**2 + slope * (p1_x - t))
            p2 = axes_left.c2p(p2_x, t**2 + slope * (p2_x - t))
            mob.set_points_as_corners([p1, p2])
        tangent.add_updater(update_tangent)
        
        # Height element (Right side)
        # Vertical segment representing the current velocity value
        height_line = Line(color=WHITE, stroke_width=4)
        def update_height(mob):
            t = t_tracker.get_value()
            p_bottom = axes_right.c2p(t, 0)
            p_top = axes_right.c2p(t, 2*t)
            mob.set_points_as_corners([p_bottom, p_top])
        height_line.add_updater(update_height)
        
        dot_vel = Dot(color=color_vel)
        dot_vel.add_updater(lambda d: d.move_to(axes_right.c2p(t_tracker.get_value(), 2*t_tracker.get_value())))

        self.play(FadeIn(dot_pos), FadeIn(tangent), FadeIn(height_line), FadeIn(dot_vel))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The integral's area is the position's vertical change."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(highlight_color)
        
        # Shaded Area under the velocity line (Right side)
        area = VMobject().set_fill(color_vel, opacity=0.4).set_stroke(width=0)
        def update_area(mob):
            t = t_tracker.get_value()
            if t <= 0.05:
                mob.set_points([])
                return
            # Polygon vertices for the trapezoid/triangle area
            pts = [
                axes_right.c2p(0, 0),
                axes_right.c2p(t, 0),
                axes_right.c2p(t, 2*t),
                axes_right.c2p(0, 0)
            ]
            mob.set_points_as_corners(pts)
        area.add_updater(update_area)
        
        # Height Change indicator (Left side) - Vertical segment from axis to curve at t
        height_change = Line(color=color_pos, stroke_width=6)
        def update_height_change(mob):
            t = t_tracker.get_value()
            p1 = axes_left.c2p(t, 0)
            p2 = axes_left.c2p(t, t**2)
            mob.set_points_as_corners([p1, p2])
        height_change.add_updater(update_height_change)
        
        self.play(FadeIn(area), FadeIn(height_change))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Watch as slope and area change in perfect sync."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(highlight_color)
        
        # Smoothly move from start to end to visualize the relationship
        self.play(t_tracker.animate.set_value(0.1), run_time=1.5)
        self.play(t_tracker.animate.set_value(2.0), run_time=4, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "This bridge connects rates of change to total accumulation."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(highlight_color)
        
        # Synchronized flash highlight to emphasize the mathematical equivalence
        self.play(
            Indicate(height_change, color=WHITE, scale_factor=1.2),
            Indicate(area, color=WHITE, scale_factor=1.05),
            Indicate(bridge_icon, color=highlight_color, scale_factor=1.5),
            run_time=2
        )
        
        self.wait(2)
