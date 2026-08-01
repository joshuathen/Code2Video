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
        # Setup the scene layout with 5 lines as per teaching script
        self.setup_layout(
            "The Time Domain vs. The Frequency Domain", 
            [
                "In the time domain, signals often look complex.", 
                "We can shift perspective to view the signal differently.", 
                "Individual frequencies emerge as distinct, clear spikes.", 
                "Each spike represents a unique ingredient of the wave.", 
                "This transformation bridges time and frequency domains."
            ]
        )

        # Colors
        GREEN_WAVE = "#00FF00"
        RED_SPIKE = "#FF0000"
        BLUE_SPIKE = "#0000FF"
        YELLOW_SPIKE = "#FFFF00"
        SHIFT_COLOR = "#00FFFF"

        # === Animation for Lecture Line 1 ===
        # Create axis for time domain (Issue 36: area C1-F6, scale 0.7)
        time_axes = Axes(
            x_range=[0, 6, 1],
            y_range=[-2, 2, 1],
            x_length=5,
            y_length=3,
            axis_config={"color": WHITE, "include_tip": True}
        )
        self.place_in_area(time_axes, 'C1', 'F6', scale_factor=0.7)
        
        # Create complex green wave: Sum of 3 sines
        wave = time_axes.plot(
            lambda t: 0.8 * np.sin(2 * PI * 0.5 * t) + 
                      0.5 * np.sin(2 * PI * 1.5 * t) + 
                      0.4 * np.sin(2 * PI * 2.5 * t),
            color=GREEN_WAVE
        )

        # Domain Labels (Issue 35: position and scale 0.7)
        time_label = Text("Time Domain", font_size=20, color=WHITE)
        self.place_at_grid(time_label, 'B2', scale_factor=0.7)
        
        freq_label = Text("Frequency Domain", font_size=20, color=WHITE)
        self.place_at_grid(freq_label, 'B5', scale_factor=0.7)
        freq_label.set_opacity(0.3)

        self.play(
            self.lecture[0].animate.set_color(GREEN_WAVE),
            Create(time_axes),
            Create(wave),
            Write(time_label),
            Write(freq_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Simulation of "Perspective Shift": Flattening the time axis
        self.play(
            self.lecture[1].animate.set_color(SHIFT_COLOR),
            wave.animate.scale(0.01, about_point=time_axes.c2p(0, 0)).set_opacity(0.2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Frequencies emerge as spikes
        freq_axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 1.5, 0.5],
            x_length=5,
            y_length=3,
            axis_config={"color": WHITE, "include_tip": True}
        )
        self.place_in_area(freq_axes, 'C1', 'F6', scale_factor=0.7)

        spike1 = Line(freq_axes.c2p(0.8, 0), freq_axes.c2p(0.8, 1.2), color=WHITE, stroke_width=6)
        spike2 = Line(freq_axes.c2p(2.0, 0), freq_axes.c2p(2.0, 0.8), color=WHITE, stroke_width=6)
        spike3 = Line(freq_axes.c2p(3.2, 0), freq_axes.c2p(3.2, 0.6), color=WHITE, stroke_width=6)
        spikes = VGroup(spike1, spike2, spike3)

        self.play(
            self.lecture[2].animate.set_color(WHITE),
            ReplacementTransform(time_axes, freq_axes),
            ReplacementTransform(wave, spikes),
            time_label.animate.set_opacity(0.3),
            freq_label.animate.set_opacity(1),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Unique ingredients (Spikes change colors)
        self.play(
            self.lecture[3].animate.set_color(RED_SPIKE),
            spike1.animate.set_color(RED_SPIKE),
            spike2.animate.set_color(BLUE_SPIKE),
            spike3.animate.set_color(YELLOW_SPIKE),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Transformation bridges domains (Swapping labels)
        target_pos_time = self.grid['B5']
        target_pos_freq = self.grid['B2']

        self.play(
            self.lecture[4].animate.set_color(YELLOW_SPIKE),
            time_label.animate.move_to(target_pos_time),
            freq_label.animate.move_to(target_pos_freq),
            run_time=2
        )
        self.wait(2)
