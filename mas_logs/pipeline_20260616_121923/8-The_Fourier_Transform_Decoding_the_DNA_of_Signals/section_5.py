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
        # Initialization
        title = "Application: Noise Cancellation"
        lines = [
            'We can use this map to edit reality.',
            'Identify and delete spikes representing unwanted noise.',
            'Transform back to hear only the clear voice.'
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_VOICE = "#00FF00"  # Green
        COLOR_NOISE = "#FF0000"  # Red
        COLOR_HIGHLIGHT = "#FFFF00" # Yellow

        # === Pre-build Mobjects ===
        
        # Time Domain Axes
        time_axes = Axes(
            x_range=[0, 4, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=4,
            y_length=2.5,
            axis_config={"include_tip": False, "color": BLUE_D}
        )
        self.place_in_area(time_axes, 'A2', 'C6', scale_factor=0.8)
        time_label = Text("Time Domain", font_size=18, color=BLUE_D)
        self.place_at_grid(time_label, 'A1', scale_factor=0.6)

        # Waves
        clean_wave = time_axes.plot(
            lambda t: 0.8 * np.sin(2 * PI * 0.5 * t),
            color=COLOR_VOICE
        )
        noisy_wave = time_axes.plot(
            lambda t: 0.8 * np.sin(2 * PI * 0.5 * t) + 0.3 * np.sin(2 * PI * 5 * t),
            color=COLOR_NOISE
        )

        # Frequency Domain Axes
        freq_axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 1.2, 0.5],
            x_length=4,
            y_length=2.5,
            axis_config={"include_tip": False, "color": PURPLE_D}
        )
        self.place_in_area(freq_axes, 'D2', 'F6', scale_factor=0.8)
        freq_label = Text("Frequency Domain", font_size=18, color=PURPLE_D)
        self.place_at_grid(freq_label, 'D1', scale_factor=0.6)

        # Spikes
        voice_spike = Line(
            freq_axes.c2p(0.5, 0), freq_axes.c2p(0.5, 0.8),
            color=COLOR_VOICE, stroke_width=8
        )
        noise_spike = Line(
            freq_axes.c2p(5, 0), freq_axes.c2p(5, 0.3),
            color=COLOR_NOISE, stroke_width=8
        )
        
        voice_spike_label = Text("Voice", font_size=14, color=COLOR_VOICE)
        self.place_at_grid(voice_spike_label, 'E3', scale_factor=0.7)
        noise_spike_label = Text("Noise", font_size=14, color=COLOR_NOISE)
        self.place_at_grid(noise_spike_label, 'E5', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        # Line: 'We can use this map to edit reality.'
        self.lecture[0].set_color(COLOR_HIGHLIGHT)
        self.play(Create(time_axes), Write(time_label))
        self.play(Create(noisy_wave), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line: 'Identify and delete spikes representing unwanted noise.'
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_HIGHLIGHT)
        
        self.play(Create(freq_axes), Write(freq_label))
        self.play(
            Create(voice_spike),
            Create(noise_spike),
            Write(voice_spike_label),
            Write(noise_spike_label)
        )
        self.wait(1)

        # Delete the noise spike
        cross = VGroup(
            Line(UP+LEFT, DOWN+RIGHT),
            Line(UP+RIGHT, DOWN+LEFT)
        ).scale(0.2).move_to(noise_spike.get_top()).set_color(RED)
        
        self.play(Create(cross))
        self.play(
            FadeOut(noise_spike),
            FadeOut(noise_spike_label),
            FadeOut(cross),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line: 'Transform back to hear only the clear voice.'
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_HIGHLIGHT)
        
        # Transition noisy wave to clean wave
        self.play(
            ReplacementTransform(noisy_wave, clean_wave),
            run_time=2
        )
        self.wait(2)

        # Reset final color
        self.lecture[2].set_color(WHITE)
        self.wait(1)
