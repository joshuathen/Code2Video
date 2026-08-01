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
        # Define Colors per instructions
        C_NOISY = "#ADD8E6"  # Light Blue
        C_CLEAN = "#00FF00"  # Green
        C_GREY = "#808080"   # Grey
        
        lines = [
            'Real-world signals often contain unwanted noise or hums.',
            'In the Frequency Domain, noise stands out clearly.',
            'We can simply delete these unwanted frequency bars.',
            'Applying an inverse transform recreates the clean signal.',
            'Now, the original signal is clear and noise-free.'
        ]
        
        self.setup_layout("Practical Application: Noise Cancellation", lines)
        
        # === Setup Objects ===
        # Signal Plot
        sig_axes = Axes(
            x_range=[0, 10, 1], y_range=[-1.5, 1.5, 1],
            x_length=5, y_length=2,
            axis_config={"include_tip": False, "color": WHITE}
        )
        sig_label = Text("Time Domain", font_size=16).next_to(sig_axes, UP, buff=0.1)
        
        noisy_curve = sig_axes.plot(
            lambda x: np.sin(x) + 0.25 * np.sin(20 * x),
            color=C_NOISY
        )
        clean_curve = sig_axes.plot(
            lambda x: np.sin(x),
            color=C_CLEAN
        )
        
        signal_group = VGroup(sig_axes, sig_label, noisy_curve)
        # Fix Issue 46: shift right and scale down to avoid overlap
        self.place_in_area(signal_group, 'B2', 'C6', scale_factor=0.9)

        # Spectrum Plot
        spec_axes = Axes(
            x_range=[0, 10, 1], y_range=[0, 2, 1],
            x_length=5, y_length=2,
            axis_config={"include_tip": False, "color": WHITE}
        )
        spec_label = Text("Frequency Domain", font_size=16).next_to(spec_axes, UP, buff=0.1)
        
        # Frequency bars
        signal_bar = Rectangle(width=0.4, height=1.5, fill_opacity=1, color=C_CLEAN, stroke_width=0)
        signal_bar.move_to(spec_axes.c2p(2, 0.75))
        
        noise_bar = Rectangle(width=0.4, height=0.6, fill_opacity=1, color=C_GREY, stroke_width=0)
        noise_bar.move_to(spec_axes.c2p(8, 0.3))
        
        spectrum_group = VGroup(spec_axes, spec_label, signal_bar, noise_bar)
        # Fix Issue 47: shift right and scale down to avoid overlap
        self.place_in_area(spectrum_group, 'E2', 'F6', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        # Real-world signals often contain unwanted noise or hums.
        self.play(self.lecture[0].animate.set_color(C_NOISY))
        self.play(Create(sig_axes), Write(sig_label))
        self.play(Create(noisy_curve), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # In the Frequency Domain, noise stands out clearly.
        self.play(self.lecture[1].animate.set_color(C_CLEAN))
        self.play(Create(spec_axes), Write(spec_label))
        self.play(FadeIn(signal_bar), FadeIn(noise_bar))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We can simply delete these unwanted frequency bars.
        self.play(self.lecture[2].animate.set_color(C_GREY))
        
        cross = VGroup(
            Line(noise_bar.get_corner(UL), noise_bar.get_corner(DR), color=RED),
            Line(noise_bar.get_corner(UR), noise_bar.get_corner(DL), color=RED)
        )
        self.play(Create(cross), run_time=0.5)
        self.play(FadeOut(noise_bar), FadeOut(cross))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Applying an inverse transform recreates the clean signal.
        self.play(self.lecture[3].animate.set_color(C_CLEAN))
        self.play(Transform(noisy_curve, clean_curve))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Now, the original signal is clear and noise-free.
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.play(Flash(noisy_curve, color=C_CLEAN, line_length=0.3))
        self.wait(2)
