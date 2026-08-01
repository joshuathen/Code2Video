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
        # Setup the scene layout
        lecture_lines = [
            'Imagine wrapping a signal around a spinning circle.',
            'We change the wrapping frequency and track the shape.',
            'Most frequencies result in a centered, messy blob.',
            'At the right frequency, the shape shifts off-center.',
            'This shift creates a peak in our frequency chart.'
        ]
        self.setup_layout("The Mechanism: The 'Wrapping' Visualization", lecture_lines)

        # Signal properties
        sig_freq = 2.0
        wrap_freq_tracker = ValueTracker(0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Pure yellow sine wave at the top
        # Using a small Axes to container the signal
        sig_axes = Axes(
            x_range=[0, 2, 0.5],
            y_range=[-1.5, 1.5, 1],
            x_length=4,
            y_length=1.5,
            axis_config={"include_tip": False, "stroke_width": 2}
        )
        # Resolved Issue 45: Reduced vertical space and scale
        self.place_in_area(sig_axes, "A1", "A6", scale_factor=0.7)
        
        sine_wave = sig_axes.plot(lambda t: np.sin(2 * PI * sig_freq * t), color="#FFFF00")
        self.play(Create(sig_axes), Create(sine_wave))
        self.wait(1)

        # Wrapping visualization setup
        wrap_center = self.grid["E3"] # Center of the wrapping area
        
        def get_wrapped_point(t, wrap_f):
            # Radius = baseline + signal amplitude
            r = 1.0 + 0.4 * np.sin(2 * PI * sig_freq * t)
            angle = -2 * PI * wrap_f * t
            return wrap_center + np.array([r * np.cos(angle), r * np.sin(angle), 0])

        wrap_curve = always_redraw(lambda: ParametricFunction(
            lambda t: get_wrapped_point(t, wrap_freq_tracker.get_value()),
            t_range=[0, 3],
            color="#FFFF00",
            stroke_width=2
        ))

        # Show the wrapping circle path (faint guide)
        guide_circle = Circle(radius=1.0, color=GRAY, stroke_opacity=0.3).move_to(wrap_center)
        
        self.play(Create(guide_circle))
        self.play(Create(wrap_curve), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        
        # Display the wrapping frequency value
        freq_label = Text("Wrap Frequency: ", font_size=24)
        # Using mob_class=Text to avoid FileNotFoundError: 'latex' when LaTeX is not installed
        freq_val = DecimalNumber(wrap_freq_tracker.get_value(), num_decimal_places=2, font_size=24, mob_class=Text)
        freq_val.add_updater(lambda d: d.set_value(wrap_freq_tracker.get_value()))
        freq_group = VGroup(freq_label, freq_val).arrange(RIGHT, buff=0.1)
        # Resolved Issue 44: Moved and scaled freq_group to avoid overlap
        self.place_at_grid(freq_group, "B5", scale_factor=0.8)
        
        self.play(Write(freq_group))
        self.play(wrap_freq_tracker.animate.set_value(1.2), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        
        # Show that at 1.2, it's a messy blob
        # Add center of mass (COM) dot
        def get_com():
            # Approximate COM by sampling
            samples = 100
            points = [get_wrapped_point(t, wrap_freq_tracker.get_value()) for t in np.linspace(0, 3, samples)]
            return np.mean(points, axis=0)

        com_dot = Dot(color="#FF0000", radius=0.08)
        com_dot.add_updater(lambda d: d.move_to(get_com()))
        
        self.play(FadeIn(com_dot))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        
        # Shift wrap frequency to match signal frequency (2.0)
        self.play(wrap_freq_tracker.animate.set_value(2.0), run_time=3)
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(YELLOW)
        
        # Show frequency chart peak
        chart_axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 1, 0.5],
            x_length=3,
            y_length=1.5,
            axis_config={"include_tip": False, "stroke_width": 2}
        )
        # Resolved Issue 43: Moved and scaled chart_axes to prevent visual crowding
        self.place_in_area(chart_axes, 'E1', 'F2', scale_factor=0.6)
        
        # The "peak" is at x=2
        peak_curve = chart_axes.plot(
            lambda x: 0.8 * np.exp(-10 * (x - 2)**2) + 0.1, 
            color=WHITE
        )
        peak_label = Text("Frequency Peak", font_size=16).next_to(chart_axes, UP, buff=0.1)
        
        self.play(Create(chart_axes), Write(peak_label))
        self.play(Create(peak_curve))
        self.wait(2)
