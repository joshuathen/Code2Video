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
        # Initialize Scene
        lecture_lines = [
            "A sprinting cheetah's speed fluctuates every second.",
            "Average speed measures distance over a time interval.",
            "But how fast is it at one exact moment?"
        ]
        self.setup_layout("The Cheetah's Paradox: Average vs. Instantaneous", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # 1. Setup Speedometer at A6 (Issue 28)
        gauge_arc = Arc(radius=0.6, start_angle=PI, angle=-PI, color=WHITE)
        label_low = Text("40", font_size=12).move_to(gauge_arc.point_from_proportion(0.2))
        label_high = Text("120", font_size=12).move_to(gauge_arc.point_from_proportion(0.8))
        kmh = Text("km/h", font_size=10).move_to(gauge_arc.get_center() + DOWN * 0.15)
        speedometer = VGroup(gauge_arc, label_low, label_high, kmh)
        self.place_at_grid(speedometer, "A6", scale_factor=0.8)
        
        # Needle pivots at arc center
        pivot = gauge_arc.get_center()
        needle = Line(pivot, pivot + UP * 0.4, color=YELLOW, buff=0)
        needle.set_stroke(width=4)

        # 2. Setup Track and Cheetah at B3-B6 (Issue 29)
        track = Line(LEFT, RIGHT, color=GRAY_D).scale(1.5) # Length 3.0 matches B3-B6 span
        cheetah = Triangle(color=ORANGE, fill_opacity=1).scale(0.12).rotate(-PI/2)
        path_group = VGroup(track, cheetah)
        self.place_in_area(path_group, 'B3', 'B6')
        cheetah.move_to(track.get_start())
        
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.add(speedometer, needle, track, cheetah)

        # Needle fluctuation tracker
        needle_tracker = ValueTracker(0.9)
        def update_needle(m):
            angle = PI * needle_tracker.get_value()
            new_end = pivot + np.array([0.4 * np.cos(angle), 0.4 * np.sin(angle), 0])
            m.put_start_and_end_on(pivot, new_end)
        needle.add_updater(update_needle)

        # Animate cheetah sprint and wild needle fluctuation
        self.play(
            cheetah.animate.move_to(track.get_end()),
            needle_tracker.animate.set_value(0.1),
            run_time=3,
            rate_func=linear
        )
        self.play(needle_tracker.animate.set_value(0.7), run_time=0.4)
        self.play(needle_tracker.animate.set_value(0.3), run_time=0.4)
        self.play(needle_tracker.animate.set_value(0.5), run_time=0.4)
        
        # === Animation for Lecture Line 2 ===
        # 1. Setup Graph at C3-F6 (Issue 27)
        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 50, 10],
            x_length=4.0,
            y_length=3.0,
            axis_config={"include_tip": True, "stroke_width": 2}
        )
        axis_labels = VGroup(
            Text("Time", font_size=14).next_to(axes.x_axis, DOWN, buff=0.1),
            Text("Dist", font_size=14).next_to(axes.y_axis, LEFT, buff=0.1)
        )
        graph_vg = VGroup(axes, axis_labels)
        self.place_in_area(graph_vg, "C3", "F6", scale_factor=0.8)
        
        # Graph curve: y = 0.5 * x^2
        curve = axes.plot(lambda x: 0.5 * x**2, x_range=[0, 10], color="#3498DB")
        
        # Average Speed Line (Secant)
        secant = DashedLine(
            axes.c2p(0, 0), 
            axes.c2p(10, 50), 
            color="#E74C3C", 
            dash_length=0.1
        )

        self.play(self.lecture[1].animate.set_color("#E74C3C"))
        self.play(FadeIn(graph_vg), Create(curve))
        self.play(Create(secant))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # 1. Instantaneous Highlight at t=5
        t_val = 5
        point_inst = Dot(axes.c2p(t_val, 12.5), color=YELLOW)
        v_line = DashedLine(
            axes.c2p(t_val, 0), 
            axes.c2p(t_val, 12.5), 
            color=WHITE, 
            stroke_width=2
        )
        
        self.play(self.lecture[2].animate.set_color("#3498DB"))
        
        # Freeze needle
        needle.clear_updaters()
        
        self.play(
            FadeIn(point_inst),
            Create(v_line),
            needle.animate.put_start_and_end_on(
                pivot, 
                pivot + np.array([0.4 * np.cos(PI*0.45), 0.4 * np.sin(PI*0.45), 0])
            )
        )
        self.wait(2)
