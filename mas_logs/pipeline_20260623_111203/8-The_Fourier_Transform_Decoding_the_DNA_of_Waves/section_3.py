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
        # Data and setup
        title_str = "Dual Perspectives: Time vs. Frequency Domain"
        lines = [
            "The time domain shows signals wiggling over time.",
            "The frequency domain reveals the hidden notes within.",
            "It is like hearing music versus seeing sheet music."
        ]
        self.setup_layout(title_str, lines)

        # Colors
        time_color = "#FFFFFF"
        freq_color = "#00FFFF"
        piano_color = "#E0E0E0"
        glow_color = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(time_color))

        # Time Domain Setup
        time_label = Text("Time Domain", font_size=24, color=time_color)
        self.place_at_grid(time_label, "A2")
        
        time_axes = Axes(
            x_range=[0, 4, 1], y_range=[-1.5, 1.5, 1],
            axis_config={"include_tip": False, "color": GREY},
            x_length=3, y_length=2
        )
        self.place_in_area(time_axes, "B1", "C3")

        # Frequency Domain Setup
        freq_label = Text("Frequency Domain", font_size=24, color=freq_color)
        self.place_at_grid(freq_label, "A5")
        
        freq_axes = Axes(
            x_range=[0, 10, 1], y_range=[0, 1.2, 0.5],
            axis_config={"include_tip": False, "color": GREY},
            x_length=3, y_length=2
        )
        self.place_in_area(freq_axes, "B4", "C6")

        # Piano visualization
        piano_keys = VGroup(*[
            Rectangle(width=0.3, height=1.0, fill_opacity=0.8, fill_color=WHITE, stroke_color=BLACK)
            for _ in range(7)
        ]).arrange(RIGHT, buff=0.05)
        self.place_in_area(piano_keys, "E2", "F5", scale_factor=0.8)

        # Single Sine Wave
        freq1 = 2
        sine_wave = time_axes.plot(lambda t: np.sin(2 * PI * freq1 * t / 4), color=time_color)

        self.play(
            FadeIn(time_label), FadeIn(time_axes),
            FadeIn(freq_label), FadeIn(freq_axes),
            Create(piano_keys)
        )
        
        self.play(
            piano_keys[2].animate.set_fill(piano_color), # Middle key highlight
            Create(sine_wave)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(freq_color)
        )

        # Single Spike
        spike1 = freq_axes.get_vertical_line(freq_axes.c2p(freq1, 1), color=freq_color)
        spike_dot = Dot(freq_axes.c2p(freq1, 1), color=freq_color, radius=0.05)
        spike_group = VGroup(spike1, spike_dot)

        self.play(Create(spike_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(glow_color)
        )

        # Three frequencies
        freq2 = 5
        freq3 = 8
        
        # Complex wave: sum of 3 sines
        complex_wave = time_axes.plot(
            lambda t: (np.sin(2 * PI * freq1 * t / 4) + 
                       0.5 * np.sin(2 * PI * freq2 * t / 4) + 
                       0.3 * np.sin(2 * PI * freq3 * t / 4)) / 1.8, 
            color=time_color
        )

        # Spikes
        spike2_line = freq_axes.get_vertical_line(freq_axes.c2p(freq2, 0.5), color=freq_color)
        spike2_dot = Dot(freq_axes.c2p(freq2, 0.5), color=freq_color, radius=0.05)
        spike3_line = freq_axes.get_vertical_line(freq_axes.c2p(freq3, 0.3), color=freq_color)
        spike3_dot = Dot(freq_axes.c2p(freq3, 0.3), color=freq_color, radius=0.05)
        
        all_spikes = VGroup(spike_group, VGroup(spike2_line, spike2_dot), VGroup(spike3_line, spike3_dot))

        self.play(
            piano_keys[0].animate.set_fill(piano_color),
            piano_keys[5].animate.set_fill(piano_color),
            Transform(sine_wave, complex_wave),
            FadeIn(all_spikes[1]),
            FadeIn(all_spikes[2])
        )
        
        # Glowing effect
        self.play(
            all_spikes.animate.set_color(glow_color),
            run_time=1
        )
        self.play(
            Indicate(all_spikes, color=glow_color, scale_factor=1.2),
            run_time=2
        )
        self.wait(2)
