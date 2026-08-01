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
        # Setup the layout with section content
        lecture_lines = [
            'Wrap the signal around a spinning circular bobbin.',
            'Most speeds result in a tangled, centered mess.',
            'At the correct frequency, the shape becomes unbalanced.',
            'The center of mass pulls away from the origin.',
            'This physical shift identifies the hidden signal frequency.'
        ]
        self.setup_layout("The Mechanism: The 'Wrapping' Trick", lecture_lines)

        # Signal and Wrapping Parameters
        sig_freq = 2.0
        wrap_freq_tracker = ValueTracker(0.3)
        visual_scale = 0.7  # scale factor for the wrapping plot area
        
        # Signal function: a DC offset (1.2) plus a cosine component (0.6)
        def signal_func(t):
            return 1.2 + 0.6 * np.cos(2 * np.pi * sig_freq * t)

        # Wrapping function converts polar (r=signal, theta=freq*t) to Cartesian
        def get_wrapped_point(t, freq):
            r = signal_func(t)
            theta = -2 * np.pi * freq * t
            return np.array([r * np.cos(theta), r * np.sin(theta), 0])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # The Coil (wrapped signal)
        # Using a fixed range of 2 seconds for visual clarity
        coil = always_redraw(lambda: ParametricFunction(
            lambda t: get_wrapped_point(t, wrap_freq_tracker.get_value()),
            t_range=[0, 2.0],
            color=WHITE,
            stroke_width=2
        ))
        
        # Background circle for origin reference
        origin_ref = Circle(radius=0.1, color=GRAY, fill_opacity=0.3)
        
        # Visual container for the wrapping trick
        # B3 to F5 area avoids the lecture text (Issue 37)
        plot_container = VGroup(coil, origin_ref)
        self.place_in_area(plot_container, "B3", "F5", scale_factor=visual_scale)
        
        self.play(Create(coil), Create(origin_ref), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Center of Mass (CoM) dot
        com_dot = Dot(color="#FFA500", radius=0.08)
        
        def update_com(mob):
            # Calculate average position of samples along the current coil
            freq = wrap_freq_tracker.get_value()
            samples = 80
            times = np.linspace(0, 2.0, samples)
            points = [get_wrapped_point(t, freq) for t in times]
            avg_point = np.mean(points, axis=0)
            # Update position relative to the plot container center
            mob.move_to(plot_container.get_center() + avg_point * visual_scale)

        com_dot.add_updater(update_com)
        self.add(com_dot)

        # Animate frequency change to show "tangled mess"
        self.play(wrap_freq_tracker.animate.set_value(1.3), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Approach the "Golden Frequency" (matching signal frequency of 2.0 Hz)
        self.play(wrap_freq_tracker.animate.set_value(2.0), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Visualize the displacement vector
        displacement_arrow = always_redraw(lambda: Arrow(
            start=plot_container.get_center(),
            end=com_dot.get_center(),
            buff=0,
            color="#FFA500",
            stroke_width=3
        ))
        self.play(Create(displacement_arrow))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Spectrum Bar on the right side (using column 6)
        # Magnitude is proportional to CoM displacement
        spectrum_bar = Rectangle(
            width=0.4, height=0.1, 
            fill_color="#FFA500", fill_opacity=0.8, stroke_width=0
        )
        # Positioning bar near the right edge
        self.place_at_grid(spectrum_bar, "E6")
        
        bar_label = Text("Frequency Peak", font_size=14, color=WHITE).next_to(spectrum_bar, DOWN, buff=0.1)

        def update_bar(mob):
            # Distance from origin in the plot
            dist = np.linalg.norm(com_dot.get_center() - plot_container.get_center())
            # Map distance to bar height
            new_height = max(0.1, dist * 3.0)
            mob.stretch_to_fit_height(new_height, about_edge=DOWN)

        spectrum_bar.add_updater(update_bar)
        
        self.play(FadeIn(spectrum_bar), Write(bar_label))
        self.wait(3)
        
        # Final wait state
        self.lecture[4].set_color(WHITE)
        self.wait(2)
