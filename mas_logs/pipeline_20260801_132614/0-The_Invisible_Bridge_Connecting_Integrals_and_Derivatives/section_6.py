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

class Section6Scene(TeachingScene):
    def construct(self):
        # Teaching Content
        lecture_lines = [
            "- A cheetah's velocity is the derivative of its position.",
            "- The area under the velocity curve is distance traveled.",
            "- We can calculate distance by integrating the velocity function.",
            "- Conversely, differentiate position to find the cheetah's speed.",
            "- These two measurements are mathematically linked."
        ]
        self.setup_layout("Real-World Application: The Racing Cheetah", lecture_lines)

        # Color Palette
        VELOCITY_COLOR = "#00FF00"
        AREA_COLOR = "#0000FF"
        POSITION_COLOR = "#FFFFFF"
        TANGENT_COLOR = "#FFFF00"

        # Asset - Issue 33
        cheetah = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/che.svg", color=VELOCITY_COLOR)
        self.place_at_grid(cheetah, "A4", scale_factor=0.6)

        # Velocity Axes and Graph (Top Half)
        ax_v = Axes(
            x_range=[0, 4.5, 1],
            y_range=[0, 3, 1],
            x_length=4.5,
            y_length=2.5,
            axis_config={"include_tip": True, "font_size": 20},
            tips=False
        )
        labels_v = ax_v.get_axis_labels(x_label="t", y_label="v(t)")
        v_func = lambda t: 0.5 * t
        v_graph = ax_v.plot(v_func, color=VELOCITY_COLOR)
        v_group = VGroup(ax_v, labels_v, v_graph)
        # Issue 44 Fix: v_group in A1 to C4
        self.place_in_area(v_group, "A1", "C4", scale_factor=0.7)

        # Position Axes and Graph (Bottom Half)
        ax_s = Axes(
            x_range=[0, 4.5, 1],
            y_range=[0, 5, 1],
            x_length=4.5,
            y_length=2.5,
            axis_config={"include_tip": True, "font_size": 20},
            tips=False
        )
        labels_s = ax_s.get_axis_labels(x_label="t", y_label="s(t)")
        s_func = lambda t: 0.25 * t**2
        s_graph = ax_s.plot(s_func, color=POSITION_COLOR)
        s_group = VGroup(ax_s, labels_s, s_graph)
        # Issue 44 Fix: s_group in D1 to F4
        self.place_in_area(s_group, "D1", "F4", scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(VELOCITY_COLOR))
        self.play(Create(ax_v), Write(labels_v), FadeIn(cheetah))
        self.play(Create(v_graph))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(AREA_COLOR))
        area = ax_v.get_area(v_graph, x_range=[0, 4], color=AREA_COLOR, opacity=0.4)
        self.play(FadeIn(area))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(AREA_COLOR))
        # Issue 45 Fix: dist_label at C5, scale 0.75
        dist_label = MathTex(r"\text{Dist} = \int_0^4 v(t) dt = 4", color=AREA_COLOR, font_size=28)
        self.place_at_grid(dist_label, "C5", scale_factor=0.75)
        self.play(Write(dist_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(POSITION_COLOR))
        self.play(Create(ax_s), Write(labels_s))
        self.play(Create(s_graph))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(TANGENT_COLOR))
        
        t_tracker = ValueTracker(2.0)
        
        # Optimized Tangent Line
        tangent_line = Line(color=TANGENT_COLOR)
        def update_tangent(line):
            t = t_tracker.get_value()
            m = 0.5 * t # s'(t)
            L = 1.0
            dx = L / np.sqrt(1 + m**2)
            x1, x2 = t - dx, t + dx
            y1, y2 = s_func(t) + m*(x1 - t), s_func(t) + m*(x2 - t)
            line.set_points_as_corners([ax_s.c2p(x1, y1), ax_s.c2p(x2, y2)])
        
        tangent_line.add_updater(update_tangent)
        
        # Synchronized Dots
        dot_v = Dot(color=TANGENT_COLOR)
        dot_v.add_updater(lambda d: d.move_to(ax_v.c2p(t_tracker.get_value(), v_func(t_tracker.get_value()))))
        
        dot_s = Dot(color=TANGENT_COLOR)
        dot_s.add_updater(lambda d: d.move_to(ax_s.c2p(t_tracker.get_value(), s_func(t_tracker.get_value()))))
        
        # slope = height labels
        slope_val = DecimalNumber(num_decimal_places=2, color=TANGENT_COLOR, font_size=24)
        slope_val.add_updater(lambda d: d.set_value(0.5 * t_tracker.get_value()))
        slope_label = MathTex(r"\text{Slope} = ", color=TANGENT_COLOR, font_size=24)
        slope_group = VGroup(slope_label, slope_val).arrange(RIGHT, buff=0.1)
        # Issue 46 Fix: slope_group at D5, scale 0.8
        self.place_at_grid(slope_group, "D5", scale_factor=0.8)
        
        height_val = DecimalNumber(num_decimal_places=2, color=TANGENT_COLOR, font_size=24)
        height_val.add_updater(lambda d: d.set_value(0.5 * t_tracker.get_value()))
        height_label = MathTex(r"\text{Height} = ", color=TANGENT_COLOR, font_size=24)
        height_group = VGroup(height_label, height_val).arrange(RIGHT, buff=0.1)
        # Issue 46 Fix: height_group at A5, scale 0.8
        self.place_at_grid(height_group, "A5", scale_factor=0.8)

        self.play(Create(tangent_line), Create(dot_v), Create(dot_s))
        self.play(Write(slope_group), Write(height_group))
        self.wait(1)
        
        # Interactive movement
        self.play(t_tracker.animate.set_value(1.0), run_time=2, rate_func=rate_functions.linear)
        self.play(t_tracker.animate.set_value(4.0), run_time=2, rate_func=rate_functions.linear)
        self.wait(2)
