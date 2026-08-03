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
        # --- Setup ---
        title = "The Hook: The Speeding Cheetah"
        lines = [
            "A cheetah sprints across the savannah at high speed.",
            "The speedometer measures its instantaneous rate of change.",
            "This specific value is known as the derivative.",
            "The total distance covered represents the accumulated change.",
            "In calculus, this total accumulation is called the integral."
        ]
        self.setup_layout(title, lines)

        # Shared value tracker for time/progress
        time_tracker = ValueTracker(0)

        # Velocity function: v(t) = 0.15*t^2 + 1 (Smooth acceleration)
        def velocity_func(t):
            return 0.15 * t**2 + 1

        # --- Animation for Lecture Line 1 ---
        # "A cheetah sprints across the savannah at high speed."
        self.lecture[0].set_color(ORANGE)
        
        # Representing cheetah with SVG [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg]
        cheetah = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        cheetah.set_color(ORANGE)
        cheetah.scale(0.4)

        # Path: Bottom grid row F1 to F6
        start_pos = self.grid["F1"]
        end_pos = self.grid["F6"]
        cheetah.move_to(start_pos)

        self.play(FadeIn(cheetah))
        # Initial run to show the cheetah moving
        self.play(cheetah.animate.move_to(end_pos), run_time=3, rate_func=slow_into)
        self.wait(0.5)

        # --- Animation for Lecture Line 2 ---
        # "The speedometer measures its instantaneous rate of change."
        self.lecture[1].set_color(WHITE)
        
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            x_length=4.5,
            y_length=3.0,
            axis_config={"color": WHITE, "include_tip": True},
            tips=False
        )
        axes_labels = axes.get_axis_labels(
            x_label=Text("Time", font_size=16), 
            y_label=Text("Speed", font_size=16)
        )
        graph_group = VGroup(axes, axes_labels)
        # Position graph in rows B1-E6 per Issue 32
        self.place_in_area(graph_group, 'B1', 'E6', scale_factor=0.8)
        
        curve = axes.plot(velocity_func, x_range=[0, 4], color=WHITE)
        
        self.play(Create(axes), Write(axes_labels))
        self.play(Create(curve))
        self.wait(1)

        # --- Animation for Lecture Line 3 ---
        # "This specific value is known as the derivative."
        self.lecture[2].set_color("#00FF00")
        
        # Reset time and cheetah for the synchronized demonstration
        time_tracker.set_value(0)
        cheetah.move_to(start_pos)
        
        # Point on the graph curve
        dot_on_graph = Dot(color="#00FF00")
        dot_on_graph.add_updater(lambda d: d.move_to(
            axes.c2p(time_tracker.get_value(), velocity_func(time_tracker.get_value()))
        ))
        
        # Highlight on the y-axis
        h_line = always_redraw(lambda: axes.get_horizontal_line(
            dot_on_graph.get_center(), color="#00FF00", stroke_width=2
        ))
        
        y_axis_dot = Dot(color="#00FF00", radius=0.06)
        y_axis_dot.add_updater(lambda d: d.move_to(
            axes.c2p(0, velocity_func(time_tracker.get_value()))
        ))
        
        # Derivative label with icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/speed.svg]
        deriv_text = Text("Derivative (Speed)", font_size=14, color="#00FF00")
        deriv_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speed.svg").set_color("#00FF00").scale(0.2)
        deriv_label = VGroup(deriv_text, deriv_icon).arrange(LEFT, buff=0.1)
        
        # Position label relative to the y-axis point
        deriv_label.add_updater(lambda l: l.next_to(y_axis_dot, LEFT, buff=0.1))

        # Sync cheetah movement with the time tracker
        cheetah.add_updater(lambda m: m.move_to(
            start_pos + (end_pos - start_pos) * (time_tracker.get_value() / 4.0)
        ))

        self.add(dot_on_graph, h_line, y_axis_dot, deriv_label)
        self.play(time_tracker.animate.set_value(4), run_time=5, rate_func=linear)
        self.wait(1)

        # --- Animation for Lecture Line 4 ---
        # "The total distance covered represents the accumulated change."
        self.lecture[3].set_color("#FFFF00")
        
        # Shade the area under the curve
        shaded_area = axes.get_area(curve, x_range=[0, 4], color="#FFFF00", opacity=0.3)
        
        self.play(FadeIn(shaded_area))
        self.wait(1)

        # --- Animation for Lecture Line 5 ---
        # "In calculus, this total accumulation is called the integral."
        self.lecture[4].set_color("#FFFF00")
        
        integral_label = Text("Integral (Distance)", font_size=18, color="#FFFF00")
        # Place label in area D4-E5 per Issue 33
        self.place_in_area(integral_label, 'D4', 'E5', scale_factor=0.7)
        
        self.play(Write(integral_label))
        self.wait(3)
