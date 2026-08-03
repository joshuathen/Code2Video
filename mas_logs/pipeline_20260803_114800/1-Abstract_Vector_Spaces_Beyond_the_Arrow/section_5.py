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
        # Setup layout with title and lecture lines
        self.setup_layout("Abstract Visualization: Function Spaces", [
            "Complex signals like sound are also vectors.",
            "Each frequency slider represents a unique dimension.",
            "A single point can represent an entire waveform."
        ])

        # Colors
        VIOLET = "#EE82EE"
        SILVER = "#C0C0C0"
        WHITE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Color the first lecture line
        self.lecture[0].set_color(VIOLET)
        
        # Define complex waveform generation
        def get_wave_points(offset):
            points = []
            # Covering range roughly matching the area
            for t in np.linspace(-2.5, 2.5, 120):
                # A sum of sines to simulate a complex sound signal
                y = 0.4 * np.sin(2 * PI * (t - offset)) + \
                    0.2 * np.sin(5 * PI * (t - offset)) + \
                    0.1 * np.sin(10 * PI * (t - offset))
                points.append([t, y, 0])
            return points

        # Persistent wave object
        wave = VMobject(color=VIOLET)
        wave.set_points_as_corners(get_wave_points(0))
        
        # Issue 31 Fix: Scale to 0.6 within A1-C6
        self.place_in_area(wave, "A1", "C6", scale_factor=0.6)
        wave_center = wave.get_center().copy()
        
        time_tracker = ValueTracker(0)
        
        def update_wave(obj):
            new_points = get_wave_points(time_tracker.get_value())
            obj.set_points_as_corners(new_points)
            obj.move_to(wave_center)

        wave.add_updater(update_wave)
        self.play(Create(wave))
        # Scroll the wave
        self.play(time_tracker.animate.set_value(2), run_time=4, rate_func=linear)
        wave.remove_updater(update_wave)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Color the second lecture line
        self.lecture[1].set_color(SILVER)
        
        # Function to create a slider with a track, knob, and label
        def create_slider(pos_label):
            track = Line(DOWN*0.6, UP*0.6, color=GRAY_D)
            knob = Dot(color=SILVER)
            label = Text(pos_label, font_size=18, color=WHITE_COLOR).set_opacity(0.4)
            slider = VGroup(track, knob, label)
            return slider

        slider_bass = create_slider("Bass")
        slider_mid = create_slider("Mid")
        slider_treble = create_slider("Treble")

        # Position sliders in row E
        self.place_at_grid(slider_bass, "E2")
        self.place_at_grid(slider_mid, "E4")
        self.place_at_grid(slider_treble, "E6")
        
        # Adjust labels to be below their tracks
        for s in [slider_bass, slider_mid, slider_treble]:
            s[2].next_to(s[0], DOWN, buff=0.2)

        # Initial knob positions
        slider_bass[1].move_to(slider_bass[0].point_from_proportion(0.3))
        slider_mid[1].move_to(slider_mid[0].point_from_proportion(0.7))
        slider_treble[1].move_to(slider_treble[0].point_from_proportion(0.5))

        self.play(
            FadeIn(slider_bass),
            FadeIn(slider_mid),
            FadeIn(slider_treble)
        )
        
        # Animation: Sliders move and labels "light up"
        self.play(
            slider_bass[1].animate.move_to(slider_bass[0].point_from_proportion(0.8)),
            slider_mid[1].animate.move_to(slider_mid[0].point_from_proportion(0.2)),
            slider_treble[1].animate.move_to(slider_treble[0].point_from_proportion(0.9)),
            slider_bass[2].animate.set_opacity(1),
            slider_mid[2].animate.set_opacity(1),
            slider_treble[2].animate.set_opacity(1),
            run_time=2
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Color the third lecture line
        self.lecture[2].set_color(WHITE_COLOR)
        
        # Create a single point representing the "state" in function space
        point_dot = Dot(color=WHITE_COLOR)
        point_label = Text("Point in Function Space", font_size=16, color=WHITE_COLOR)
        point_group = VGroup(point_dot, point_label).arrange(UP, buff=0.2)
        
        # Issue 30 Fix: Scale to 0.8 within D1-D6
        self.place_in_area(point_group, "D1", "D6", scale_factor=0.8)
        
        # Show the point group
        self.play(FadeIn(point_group))
        
        # Highlight the concept: point = waveform
        self.play(Indicate(point_dot, color=WHITE_COLOR, scale_factor=1.4))
        self.play(Indicate(wave, color=WHITE_COLOR, scale_factor=1.1))
        
        self.wait(2)
