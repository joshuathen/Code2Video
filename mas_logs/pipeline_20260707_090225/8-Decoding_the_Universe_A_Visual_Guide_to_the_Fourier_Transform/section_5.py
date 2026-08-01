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
        # Setup the scene with title and lecture lines
        lecture_lines = [
            "Felix's audio has a constant, annoying high-pitched hum.",
            "We find and delete the hum's specific frequency spike.",
            "Converting back leaves us with perfectly clean sound."
        ]
        self.setup_layout("Application: Felix the Cat and the Noisy Room", lecture_lines)

        # Colors
        GREY_COLOR = "#808080"
        GREEN_COLOR = "#00FF00"
        SPIKE_COLOR = "#FF0000" # Highlighting the "bad" hum

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(GREY_COLOR))
        
        # Create Time-Domain Axes
        time_axes = Axes(
            x_range=[0, 4, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=5,
            y_length=3,
            axis_config={"include_tip": False, "color": WHITE},
            tips=False
        )
        time_label = Text("Time Domain", font_size=18, color=WHITE)
        time_group = VGroup(time_axes, time_label)
        time_label.next_to(time_axes, UP, buff=0.1)
        # Resolved Issue 38: Moving time_group down to avoid crowding the title
        self.place_in_area(time_group, "B1", "C6", scale_factor=0.5)

        # Define waves
        def noisy_func(x):
            return 0.7 * np.sin(2 * PI * 1.0 * x) + 0.3 * np.sin(2 * PI * 8.0 * x)
        
        def clean_func(x):
            return 0.7 * np.sin(2 * PI * 1.0 * x)

        noisy_wave = time_axes.plot(noisy_func, color=GREY_COLOR, x_range=[0, 4])
        
        self.play(Create(time_axes), Write(time_label))
        self.play(Create(noisy_wave))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(SPIKE_COLOR))

        # Create Frequency-Domain Axes
        freq_axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 1.2, 0.5],
            x_length=5,
            y_length=3,
            axis_config={"include_tip": False, "color": WHITE},
            tips=False
        )
        freq_label = Text("Frequency Spectrum", font_size=18, color=WHITE)
        freq_group = VGroup(freq_axes, freq_label)
        freq_label.next_to(freq_axes, UP, buff=0.1)
        # Resolved Issue 39: Moving freq_group down to improve separation
        self.place_in_area(freq_group, "E1", "F6", scale_factor=0.5)

        # Create Spikes
        # Low frequency component (signal)
        signal_spike = Line(
            freq_axes.c2p(1, 0), freq_axes.c2p(1, 0.7),
            color=WHITE, stroke_width=4
        )
        # High frequency component (the "hum")
        hum_spike = Line(
            freq_axes.c2p(8, 0), freq_axes.c2p(8, 0.3),
            color=SPIKE_COLOR, stroke_width=4
        )
        hum_label = Text("Hum", font_size=14, color=SPIKE_COLOR).next_to(hum_spike, UP, buff=0.05)

        self.play(Create(freq_axes), Write(freq_label))
        self.play(Create(signal_spike), Create(hum_spike), Write(hum_label))
        self.wait(1)

        # Delete the hum spike
        self.play(
            FadeOut(hum_spike),
            FadeOut(hum_label),
            run_time=1
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN_COLOR))

        # Transform noisy wave to clean wave
        clean_wave = time_axes.plot(clean_func, color=GREEN_COLOR, x_range=[0, 4])
        
        self.play(
            Transform(noisy_wave, clean_wave),
            run_time=2
        )
        self.wait(2)
