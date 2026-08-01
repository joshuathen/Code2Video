from manim import *

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
        self.setup_layout("Time Domain vs. Frequency Domain", [
            "- Time domain shows signals changing over time.",
            "- Frequency domain reveals the individual frequencies present.",
            "- Fourier Transform bridges these two distinct perspectives."
        ])
        
        # === Animation for Lecture Line 1 ===
        # A messy magenta signal wave (#FF00FF) oscillates in a 2D time-plot.
        
        axes_time = Axes(
            x_range=[0, 4, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=3,
            axis_config={"color": WHITE, "include_tip": True},
            tips=False
        )
        time_label = Text("Time", font_size=16, color=WHITE)
        amp_label = Text("Amp", font_size=16, color=WHITE)
        
        # Wave: Sum of 3 sines to look "messy"
        wave = axes_time.plot(
            lambda t: np.sin(2 * PI * 1 * t) + 0.5 * np.sin(2 * PI * 3 * t) + 0.2 * np.sin(2 * PI * 5 * t),
            color="#FF00FF",
            x_range=[0, 4]
        )
        
        plot_group_time = VGroup(axes_time, wave)
        self.place_in_area(plot_group_time, 'B1', 'E5', scale_factor=0.8)
        
        # Fixed: issue 28 (place in area E5-E6)
        self.place_in_area(time_label, 'E5', 'E6', scale_factor=0.7)
        time_label.shift(DOWN * 0.4) # Adjust for axis alignment
        
        self.place_at_grid(amp_label, 'B1', scale_factor=0.8)
        amp_label.shift(UP * 0.3 + LEFT * 0.3)
        
        self.play(self.lecture[0].animate.set_color("#FF00FF"))
        self.play(Create(axes_time), Create(wave), Write(time_label), Write(amp_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The view rotates 90 degrees to show three sharp yellow spikes (#FFFF00) on a frequency-axis.
        
        axes_freq = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 1.5, 0.5],
            x_length=4,
            y_length=3,
            axis_config={"color": WHITE, "include_tip": True},
            tips=False
        )
        freq_label = Text("Frequency", font_size=16, color=WHITE)
        mag_label = Text("Mag", font_size=16, color=WHITE)
        
        # Spikes at 1, 3, 5 corresponding to the sines used in the wave
        s1 = Line(axes_freq.c2p(1, 0), axes_freq.c2p(1, 1), color="#FFFF00", stroke_width=4)
        s2 = Line(axes_freq.c2p(3, 0), axes_freq.c2p(3, 0.5), color="#FFFF00", stroke_width=4)
        s3 = Line(axes_freq.c2p(5, 0), axes_freq.c2p(5, 0.2), color="#FFFF00", stroke_width=4)
        spikes = VGroup(s1, s2, s3)
        
        plot_group_freq = VGroup(axes_freq, spikes)
        self.place_in_area(plot_group_freq, 'B1', 'E5', scale_factor=0.8)
        
        # Fixed: issue 27 (place in area E5-E6)
        self.place_in_area(freq_label, 'E5', 'E6', scale_factor=0.7)
        freq_label.shift(DOWN * 0.4)
        
        self.place_at_grid(mag_label, 'B1', scale_factor=0.8)
        mag_label.shift(UP * 0.3 + LEFT * 0.3)

        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        # Simulated rotation/perspective shift via Transform
        self.play(
            ReplacementTransform(wave, spikes),
            ReplacementTransform(axes_time, axes_freq),
            ReplacementTransform(time_label, freq_label),
            ReplacementTransform(amp_label, mag_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A cyan double-headed arrow (#00FFFF) pulses between the time-wave and the frequency-spikes.
        
        arrow = DoubleArrow(
            start=self.grid['F2'],
            end=self.grid['F5'],
            color="#00FFFF",
            stroke_width=6
        )
        
        ft_text = Text("Fourier Transform", font_size=20, color="#00FFFF")
        
        # Fixed: issue 26 (place in area F2-F5)
        self.place_in_area(ft_text, 'F2', 'F5', scale_factor=0.8)
        ft_text.shift(UP * 0.4) # Shift up to avoid overlapping with arrow
        
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.play(GrowFromCenter(arrow), Write(ft_text))
        
        # Pulse the arrow to emphasize the bridge
        for _ in range(2):
            self.play(arrow.animate.scale(1.1), run_time=0.4, rate_func=there_and_back)
        
        self.wait(2)
