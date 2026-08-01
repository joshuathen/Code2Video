from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        lecture_texts = []
        for line in lecture_lines:
            if line and str(line).strip():
                # Changing Text to Tex to bypass environment-specific Pango/Cairo ParseErrors
                lecture_texts.append(Tex(str(line), font_size=22, color=WHITE))
            else:
                # Add a dummy object to maintain indexing if line is empty
                lecture_texts.append(Dot(radius=0, fill_opacity=0))
        # Base setup
        self.camera.background_color = "#000000"
        
        # Title logic: Ensuring non-empty string and single addition
        clean_title = str(title_text) if title_text and str(title_text).strip() else "Mechanism"
        self.title = Text(clean_title, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content: Ensure no empty strings passed to Text
        lecture_texts = []
        for line in lecture_lines:
            if line and str(line).strip():
                lecture_texts.append(Text(str(line), font_size=22, color=WHITE))
            else:
                # Add a dummy object to maintain indexing if line is empty
                lecture_texts.append(Dot(radius=0, fill_opacity=0))
        
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]
        cols = ["1", "2", "3", "4", "5", "6"]

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
        center = (tl_pos + br_pos) / 2
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section4Scene(TeachingScene):
    def construct(self):
        # 1. Setup Layout
        title_text = "The Mechanism: The 'Wrapping' Visualization"
        lecture_lines = [
            "Wrap signal around circle.", 
            "Center stays near origin.", 
            "Match frequency, things change.", 
            "Signal bunches, pulling center.", 
            "Shift creates chart peak."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        PINK_COLOR = "#FF00FF"
        WHITE_COLOR = "#FFFFFF"
        YELLOW_COLOR = "#FFFF00"
        
        # Positioning center using reference area
        origin_ref = Dot(radius=0, fill_opacity=0)
        self.place_in_area(origin_ref, "A1", "F6")
        center_pos = origin_ref.get_center()

        # Signal parameters
        SIGNAL_FREQ = 2.0
        BIAS = 1.2
        f_wind_tracker = ValueTracker(0.5)
        
        def get_wrapped_points(f_wind_val):
            # Sample density for smooth curves
            t_vals = np.linspace(0, 4.0, 600)
            s_t = (np.cos(2 * np.pi * SIGNAL_FREQ * t_vals) + BIAS) * 0.7
            # Polar conversion centered at center_pos (Clockwise winding)
            x = center_pos[0] + s_t * np.cos(-2 * np.pi * f_wind_val * t_vals)
            y = center_pos[1] + s_t * np.sin(-2 * np.pi * f_wind_val * t_vals)
            return [np.array([px, py, 0]) for px, py in zip(x, y)]

        # The wrapped wire (signal)
        wrapped_signal = VMobject(color=PINK_COLOR, stroke_width=2.5)
        wrapped_signal.set_points_as_corners(get_wrapped_points(f_wind_tracker.get_value()))
        
        # Updaters
        wrapped_signal.add_updater(lambda m: m.set_points_as_corners(get_wrapped_points(f_wind_tracker.get_value())))

        # Center of Mass dot
        com_dot = Dot(color=WHITE_COLOR, radius=0.08)
        com_dot.add_updater(lambda m: m.move_to(np.mean(wrapped_signal.get_points(), axis=0)))

        # Peak Vector Arrow (Yellow)
        peak_vector = Arrow(
            start=center_pos, 
            end=center_pos + UP*0.1, 
            buff=0, 
            color=YELLOW_COLOR, 
            stroke_width=4,
            max_tip_length_to_length_ratio=0.25
        )
        peak_vector.add_updater(lambda m: m.put_start_and_end_on(center_pos, com_dot.get_center()))

        # Animation Sequence
        # Line 1: Wrap signal around circle
        self.play(self.lecture[0].animate.set_color(PINK_COLOR))
        self.play(Create(wrapped_signal), run_time=3)
        self.wait(1)

        # Line 2: Center stays near origin
        self.play(self.lecture[1].animate.set_color(WHITE_COLOR))
        self.play(FadeIn(com_dot))
        self.wait(2)

        # Line 3: Match frequency
        self.play(self.lecture[2].animate.set_color(PINK_COLOR))
        self.play(f_wind_tracker.animate.set_value(1.4), run_time=3, rate_func=linear)
        self.wait(0.5)

        # Line 4: Signal bunches, pulling center
        self.play(self.lecture[3].animate.set_color(YELLOW_COLOR))
        self.play(FadeIn(peak_vector))
        self.play(f_wind_tracker.animate.set_value(SIGNAL_FREQ), run_time=2.5)
        self.wait(2)

        # Line 5: Shift creates chart peak
        self.play(self.lecture[4].animate.set_color(YELLOW_COLOR))
        self.play(Indicate(peak_vector, color=YELLOW_COLOR, scale_factor=1.15), run_time=1.5)
        self.wait(3)