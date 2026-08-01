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
        # Define lecture lines
        lecture_lines = [
            "Imagine wrapping our signal around a central circular origin.",
            "We spin the signal at different test frequencies.",
            "If frequencies don't match, the shape stays centered.",
            "When they match, the center of mass shifts dramatically.",
            "This shift flags a hidden frequency in our signal."
        ]
        self.setup_layout("The Winding Machine: Visualizing the Math", lecture_lines)
        
        # Define colors from storyboard
        signal_color = "#FFFFFF"
        circle_color = "#808080"
        vector_color = "#00FFFF"
        com_color = "#FF00FF"

        # Pre-calculate winding origin to be centered in D2-F5
        # Addressing Issue 33/34: Providing enough vertical space and horizontal buffer
        d2 = self.grid["D2"]
        f5 = self.grid["F5"]
        winding_origin = (d2 + f5) / 2

        # === Animation for Lecture Line 1 ===
        # Imagine wrapping our signal around a central circular origin.
        # Display a white #FFFFFF signal wave.
        self.lecture[0].set_color(signal_color)
        
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[-1.2, 1.2, 1],
            x_length=3.5,
            y_length=1.2,
            axis_config={"include_tip": False, "stroke_width": 2, "color": GRAY}
        )
        
        f_signal = 2.0  # 2Hz signal
        signal_graph = axes.plot(lambda t: np.cos(2 * np.pi * f_signal * t), color=signal_color)
        signal_label = Text("Signal f(t)", color=signal_color, font_size=18)
        
        signal_group = VGroup(axes, signal_graph)
        # Fix: Addressing Issue 33/34 by placing signal_group in a clearer area
        # and ensuring it doesn't crowd the winding plot.
        self.place_in_area(signal_group, "B2", "C5", scale_factor=0.7)
        signal_label.next_to(signal_group, UP, buff=0.1)
        
        self.play(Create(axes), Create(signal_graph), Write(signal_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # We spin the signal at different test frequencies.
        # Draw a gray #808080 circle below the wave.
        self.lecture[1].set_color(circle_color)
        
        circle = Circle(radius=0.9, color=circle_color, stroke_width=2).move_to(winding_origin)
        circle_label = Text("Winding Plane", color=circle_color, font_size=16)
        circle_label.next_to(circle, UP, buff=0.1)
        
        self.play(Create(circle), Write(circle_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # If frequencies don't match, the shape stays centered.
        # Wrap the signal around the circle in white #FFFFFF.
        self.lecture[2].set_color(signal_color)
        
        f_test = ValueTracker(0.5) # Start with a mismatching frequency
        
        def get_wrapped_signal_path(f):
            # Complex number representation: (f(t) + offset) * e^(-i * 2pi * f_test * t)
            return ParametricFunction(
                lambda t: winding_origin + np.array([
                    (np.cos(2 * np.pi * f_signal * t) + 1.2) * np.cos(-2 * np.pi * f * t) * 0.35,
                    (np.cos(2 * np.pi * f_signal * t) + 1.2) * np.sin(-2 * np.pi * f * t) * 0.35,
                    0
                ]),
                t_range=[0, 4, 0.05],
                color=signal_color,
                stroke_width=2
            )
            
        wrapped_signal = always_redraw(lambda: get_wrapped_signal_path(f_test.get_value()))
        
        # Display the wrapping frequency using DecimalNumber to follow L011/L014
        freq_value = DecimalNumber(f_test.get_value(), num_decimal_places=2, font_size=16, color=WHITE, mob_class=Text)
        freq_value.add_updater(lambda m: m.set_value(f_test.get_value()))
        freq_label_text = Text("Test Frequency: ", font_size=16, color=WHITE)
        freq_unit = Text(" Hz", font_size=16, color=WHITE)
        freq_group = VGroup(freq_label_text, freq_value, freq_unit).arrange(RIGHT, buff=0.1)
        freq_group.next_to(circle, DOWN, buff=0.2)
        
        self.play(Create(wrapped_signal), Write(freq_group))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # When they match, the center of mass shifts dramatically.
        # Add a cyan #00FFFF vector rotating at a variable frequency.
        self.lecture[3].set_color(vector_color)
        
        t_curr = ValueTracker(0)
        # Vector pointing to the current point on the winding curve
        vector = always_redraw(lambda: Arrow(
            start=winding_origin,
            end=winding_origin + np.array([
                (np.cos(2 * np.pi * f_signal * t_curr.get_value()) + 1.2) * np.cos(-2 * np.pi * f_test.get_value() * t_curr.get_value()) * 0.35,
                (np.cos(2 * np.pi * f_signal * t_curr.get_value()) + 1.2) * np.sin(-2 * np.pi * f_test.get_value() * t_curr.get_value()) * 0.35,
                0
            ]),
            buff=0,
            color=vector_color,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.15
        ))
        
        self.add(vector)
        # Animate the vector spinning while we change frequency to resonance
        self.play(
            t_curr.animate.set_value(4), 
            f_test.animate.set_value(2.0), 
            run_time=4, 
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This shift flags a hidden frequency in our signal.
        # Show a magenta #FF00FF dot at the shape's center of mass.
        self.lecture[4].set_color(com_color)
        
        def get_com_pos(f):
            ts = np.linspace(0, 4, 100)
            xs = (np.cos(2 * np.pi * f_signal * ts) + 1.2) * np.cos(-2 * np.pi * f * ts) * 0.35
            ys = (np.cos(2 * np.pi * f_signal * ts) + 1.2) * np.sin(-2 * np.pi * f * ts) * 0.35
            return winding_origin + np.array([np.mean(xs), np.mean(ys), 0])

        com_dot = always_redraw(lambda: Dot(get_com_pos(f_test.get_value()), color=com_color, radius=0.08))
        # Fix: Addressing Issue 34 by using a smaller font and safer placement (UP) for the COM label
        com_label = Text("Center of Mass", font_size=14, color=com_color)
        com_label.add_updater(lambda m: m.next_to(com_dot, UP, buff=0.1))

        self.play(FadeIn(com_dot), Write(com_label))
        self.wait(1)
        
        # Demonstrate the shift again by sweeping frequency away and back
        self.play(f_test.animate.set_value(0.7), run_time=2)
        self.wait(1)
        self.play(f_test.animate.set_value(2.0), run_time=2)
        
        self.wait(2)
