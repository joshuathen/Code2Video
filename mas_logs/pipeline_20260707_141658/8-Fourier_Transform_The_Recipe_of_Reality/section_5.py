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
        # Data from shared state
        title_text = "Practical Magic: Noise Cancellation"
        lecture_lines = [
            "Noise appears as unwanted spikes in frequency.",
            "Setting these specific spikes to zero removes noise.",
            "Transforming back results in a clean, clear signal."
        ]
        
        # Initialize layout
        self.setup_layout(title_text, lecture_lines)
        
        # Define colors as per storyboard and matching requirement
        VOICE_COLOR = "#00FF00"  # Green
        NOISE_COLOR = "#FF0000"  # Red
        
        # === Animation for Lecture Line 1 ===
        # Line 1 focus: Noise (Red)
        self.lecture[0].set_color(NOISE_COLOR)
        
        # Microphone asset
        mic_asset_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/microphone.svg"
        microphone = SVGMobject(mic_asset_path, color=WHITE)
        self.place_at_grid(microphone, "B1", scale_factor=0.6)
        
        # Time domain axes in top area
        ax_time = Axes(
            x_range=[0, 4, 1],
            y_range=[-1.5, 1.5, 1],
            axis_config={"include_tip": False, "color": BLUE_D},
            x_length=5,
            y_length=2.5
        )
        # Apply VideoCritic fix: scale_factor=0.8
        self.place_in_area(ax_time, "A1", "C6", scale_factor=0.8)
        time_label = Text("Time Domain Signal", font_size=18).next_to(ax_time, UP, buff=0.1)
        
        # Create voice (green) and noise (red) waves
        # Frequency 0.5 Hz for voice, 5 Hz for noise
        voice_wave = ax_time.plot(lambda x: np.sin(2 * np.pi * 0.5 * x), color=VOICE_COLOR)
        noise_wave = ax_time.plot(lambda x: 0.3 * np.sin(2 * np.pi * 5 * x), color=NOISE_COLOR)
        
        self.play(FadeIn(microphone), Create(ax_time), Write(time_label))
        self.play(Create(voice_wave), Create(noise_wave), run_time=2)
        self.wait(0.5)
        
        # Combine them into a single white signal
        combined_wave = ax_time.plot(
            lambda x: np.sin(2 * np.pi * 0.5 * x) + 0.3 * np.sin(2 * np.pi * 5 * x), 
            color=WHITE
        )
        self.play(
            ReplacementTransform(VGroup(voice_wave, noise_wave), combined_wave),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2 focus: Noise Removal (Red)
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(NOISE_COLOR)
        
        # Frequency domain axes in bottom area
        ax_freq = Axes(
            x_range=[0, 8, 1],
            y_range=[0, 1.2, 0.5],
            axis_config={"include_tip": False, "color": BLUE_D},
            x_length=5,
            y_length=2.5
        )
        # Apply VideoCritic fix: scale_factor=0.8
        self.place_in_area(ax_freq, "D1", "F6", scale_factor=0.8)
        freq_label = Text("Frequency Spectrum", font_size=18).next_to(ax_freq, UP, buff=0.1)
        
        # Frequency spikes corresponding to the time-domain components
        voice_spike = Line(
            ax_freq.c2p(0.5, 0), ax_freq.c2p(0.5, 1.0), 
            color=VOICE_COLOR, stroke_width=6
        )
        noise_spike = Line(
            ax_freq.c2p(5.0, 0), ax_freq.c2p(5.0, 0.3), 
            color=NOISE_COLOR, stroke_width=6
        )
        
        self.play(Create(ax_freq), Write(freq_label))
        self.play(GrowFromEdge(voice_spike, DOWN), GrowFromEdge(noise_spike, DOWN))
        self.wait(1)
        
        # Demonstrate noise cancellation by zeroing the noise spike
        self.play(noise_spike.animate.scale(0.001, about_edge=DOWN), run_time=1.5)
        self.remove(noise_spike)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3 focus: Clean Signal (Green)
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(VOICE_COLOR)
        
        # Show the result of filtering in the time domain
        clean_voice_wave = ax_time.plot(lambda x: np.sin(2 * np.pi * 0.5 * x), color=VOICE_COLOR)
        
        self.play(
            ReplacementTransform(combined_wave, clean_voice_wave),
            run_time=2
        )
        self.wait(2)
