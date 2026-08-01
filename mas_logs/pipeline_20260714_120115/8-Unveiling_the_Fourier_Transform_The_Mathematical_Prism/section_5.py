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
        # Data preparation from storyboard
        title_text = "Real-World Magic: Applications"
        lecture_lines = [
            "We use this math to compress large digital files.",
            "It also helps filter out unwanted background noise.",
            "By removing specific frequencies, we clarify the important data."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Visual elements setup
        # Assets integration (Issue 26)
        microphone = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/microphone.svg")
        file_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/file.svg")
        
        # Avoid column 1 (Lesson L010)
        self.place_at_grid(microphone, "C2", scale_factor=0.6)
        self.place_at_grid(file_icon, "E6", scale_factor=0.6)

        # Create an axis system for the signal visualization
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=5,
            y_length=4,
            axis_config={"color": BLUE_E, "include_tip": False, "include_ticks": False},
        )
        self.place_in_area(axes, "B2", "E6", scale_factor=0.8)

        # Define the signals
        # Base smooth signal (White)
        smooth_signal = axes.plot(lambda x: np.sin(x * PI), color="#FFFFFF")
        # High frequency noise (Gray)
        jitter_signal = axes.plot(lambda x: 0.2 * np.sin(20 * x * PI), color="#808080")
        # Combined "Noisy" signal
        noisy_signal = axes.plot(lambda x: np.sin(x * PI) + 0.2 * np.sin(20 * x * PI), color="#FFFFFF")

        # Labels - applying VideoCritic fixes (Issues 38, 39, 40)
        signal_label = Text("Original Signal", font_size=18, color=WHITE)
        self.place_in_area(signal_label, 'A2', 'A3', scale_factor=0.8) # Fix Issue 38
        
        noise_label = Text("Noise (High Frequency)", font_size=18, color="#808080")
        self.place_in_area(noise_label, 'A4', 'A6', scale_factor=0.7) # Fix Issue 39

        clean_label = Text("Filtered Signal", font_size=18, color="#00FFFF")
        # Pre-place clean_label for later transform, but keep it hidden
        self.place_in_area(clean_label, 'A2', 'A3', scale_factor=0.8) # Fix Issue 40

        # === Animation for Lecture Line 1 ===
        # Display a white #FFFFFF signal with gray #808080 jitter originating from a microphone.
        self.play(
            self.lecture[0].animate.set_color(YELLOW),
            FadeIn(microphone),
            Create(axes),
            Create(noisy_signal),
            Write(signal_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight and remove the gray jitter components.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW),
            run_time=0.5
        )
        
        # Transition from the single 'noisy' line to distinct smooth and jitter lines
        self.remove(noisy_signal)
        self.add(smooth_signal, jitter_signal)
        
        self.play(
            jitter_signal.animate.set_color(RED), # Highlighting
            Write(noise_label),
            run_time=1
        )
        self.play(
            FadeOut(jitter_signal),
            FadeOut(noise_label),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Reveal a clean, smooth cyan #00FFFF signal and save it to a file.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW),
            smooth_signal.animate.set_color("#00FFFF"),
            Transform(signal_label, clean_label),
            FadeIn(file_icon),
            run_time=2
        )
        
        # Save animation: Move a copy of the signal towards the file icon
        save_signal = smooth_signal.copy()
        self.play(
            save_signal.animate.scale(0.1).move_to(file_icon.get_center()),
            FadeOut(save_signal),
            run_time=1
        )
        
        self.wait(2)
