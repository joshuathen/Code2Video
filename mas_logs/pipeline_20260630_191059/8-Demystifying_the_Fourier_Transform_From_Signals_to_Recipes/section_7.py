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

class Section7Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        self.setup_layout(
            "Real-World Application: Noise Cancellation",
            [
                "Noise cancellation uses this recipe to identify unwanted sounds.",
                "Computers pluck out the specific frequencies causing noise.",
                "The result is pure music without the background hum."
            ]
        )

        # Define Colors
        CLEAN_COLORS = ["#3498DB", "#E74C3C", "#F1C40F"]  # Blue, Red, Yellow
        NOISE_COLOR = "#95A5A6" # Grey
        HIGHLIGHT_COLOR = "#2ECC71" # Green for selector

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Create Frequency Plot Axes using Grid System
        # x-axis from E2 to E6
        x_axis = Arrow(start=self.grid["E2"] + LEFT*0.2, end=self.grid["E6"] + RIGHT*0.5, color=WHITE, buff=0)
        # y-axis from E2 to A2
        y_axis = Arrow(start=self.grid["E2"] + DOWN*0.2, end=self.grid["A2"] + UP*0.2, color=WHITE, buff=0)
        
        # Labels using grid positioning (Resolving Issue 62)
        frequency_label = Text("Frequency", font_size=18)
        self.place_in_area(frequency_label, "F2", "F6", scale_factor=0.8)
        
        amplitude_label = Text("Amplitude", font_size=18).rotate(90*DEGREES)
        self.place_at_grid(amplitude_label, "C1", scale_factor=0.7)
        
        axes_group = VGroup(x_axis, y_axis, frequency_label, amplitude_label)

        # Create Clean Spikes (positioned at E3, E4, E5)
        clean_spikes = VGroup()
        spike_cols = ["3", "4", "5"]
        spike_heights = [2.5, 1.8, 2.2]
        for col, height, color in zip(spike_cols, spike_heights, CLEAN_COLORS):
            base_point = self.grid[f"E{col}"]
            top_point = base_point + UP * height
            line = Line(base_point, top_point, color=color, stroke_width=6)
            clean_spikes.add(line)

        # Create Noise Spikes (many small grey spikes)
        noise_spikes = VGroup()
        np.random.seed(42) # Deterministic randomness
        for i in range(25):
            # Distribute noise spikes across the plot (Cols 2 to 6)
            offset_x = np.random.uniform(0, 4.0)
            start_pos = self.grid["E2"] + RIGHT * offset_x
            height = np.random.uniform(0.1, 0.6)
            line = Line(start_pos, start_pos + UP * height, color=NOISE_COLOR, stroke_width=2)
            noise_spikes.add(line)

        self.play(
            Create(axes_group),
            Create(clean_spikes),
            Create(noise_spikes),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)

        # Selector Box to highlight the noise area (Resolving Issue 60 and 61)
        selector_box = Rectangle(
            width=4.5, 
            height=0.8, 
            color=HIGHLIGHT_COLOR, 
            stroke_width=2,
            fill_opacity=0.1
        )
        # Using E2 to E6 as requested by Issue 61, and scale factor for Issue 60
        self.place_in_area(selector_box, "E2", "E6", scale_factor=1.0)
        # Shift slightly up so it's not exactly on the axis line
        selector_box.shift(UP * 0.4)

        self.play(Create(selector_box), run_time=1)
        self.wait(1)

        # "Pluck out" the frequencies (Fade out noise and selector)
        self.play(
            FadeOut(noise_spikes),
            FadeOut(selector_box),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color(WHITE)

        # Show the "Pure music" by slightly emphasizing the clean spikes
        self.play(
            clean_spikes.animate.set_stroke(width=10),
            run_time=1
        )
        self.play(
            clean_spikes.animate.set_stroke(width=6),
            run_time=1
        )
        
        self.wait(2)
