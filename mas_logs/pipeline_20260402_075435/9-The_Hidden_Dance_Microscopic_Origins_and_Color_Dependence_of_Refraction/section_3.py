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
        # Initialize Scene
        title = "Origin: Phase Lag and Superposition"
        lines = [
            "Vibrating electrons emit their own secondary electromagnetic waves.",
            "These secondary waves are slightly delayed from the original.",
            "Interference between waves creates a new, total wave.",
            "This combined wave appears to move slower through matter.",
            "Phase lag is the true origin of refractive index."
        ]
        self.setup_layout(title, lines)

        # Constants for waves
        WAVE_X_RANGE = [0, 4.5]
        WAVE_AMPLITUDE = 0.6
        WAVE_FREQUENCY = 1.0
        
        # ValueTracker for animation time
        time_tracker = ValueTracker(0)
        self.add(time_tracker)
        time_tracker.add_updater(lambda dt, dt_val: dt.increment_value(dt_val))

        # === Animation for Lecture Line 1 ===
        # Draw a white sine wave (#FFFFFF) moving right. Place an electron oscillator [Asset: ...] in its path.
        self.lecture[0].set_color(WHITE)
        
        # Define the wave function: y = A * sin(k*x - omega*t)
        def get_white_wave():
            return FunctionGraph(
                lambda x: WAVE_AMPLITUDE * np.sin(2 * PI * (x - time_tracker.get_value())),
                x_range=WAVE_X_RANGE,
                color=WHITE
            )

        white_wave = get_white_wave()
        white_wave.add_updater(lambda m: m.become(get_white_wave()))
        
        # Create a group for placement reference
        wave_group = VGroup(white_wave)
        # Issue 35: Place wave group in B1-D6 area
        self.place_in_area(wave_group, "B1", "D6", scale_factor=0.9)
        
        # Electron Oscillator [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/electron.svg]
        electron = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/electron.svg")
        electron.set_color(BLUE).scale(0.3)
        electron_label = Text("e-", font_size=18, color=BLUE)
        electron_group = VGroup(electron, electron_label)
        
        # Position helpers based on the grid
        electron_x_grid = self.grid["C3"][0]
        electron_base_y = self.grid["C3"][1]
        
        # Keep label relative to electron
        electron_label.next_to(electron, UP, buff=0.1)

        def update_electron(m):
            t = time_tracker.get_value()
            # Calculate oscillation based on white wave phase at electron_x_grid
            # Align local coordinates with the graph offset
            local_x = electron_x_grid - wave_group.get_left()[0]
            oscillation = WAVE_AMPLITUDE * np.sin(2 * PI * (local_x - t))
            m.move_to([electron_x_grid, electron_base_y + oscillation, 0])

        electron_group.add_updater(update_electron)

        self.add(wave_group, electron_group)
        self.play(Create(white_wave), FadeIn(electron_group), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Generate a cyan sine wave (#00FFFF) originating from the electron, phase-shifted by 90 degrees.
        self.lecture[1].set_color("#00FFFF")
        
        phase_lag = PI / 2  # 90 degrees delay
        
        def get_cyan_wave():
            return FunctionGraph(
                lambda x: WAVE_AMPLITUDE * 0.7 * np.sin(2 * PI * (x - time_tracker.get_value()) - phase_lag),
                x_range=WAVE_X_RANGE,
                color="#00FFFF"
            )

        cyan_wave = get_cyan_wave()
        cyan_wave.add_updater(lambda m: m.become(get_cyan_wave()))
        
        # Match position with white wave
        cyan_wave.move_to(white_wave.get_center())
        
        self.play(Create(cyan_wave), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate both waves existing simultaneously, with the cyan wave slightly behind the white wave.
        self.lecture[2].set_color(WHITE) 
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # Combine them into a single green sine wave (#00FF00). Show its peaks trailing the original white wave peaks.
        self.lecture[3].set_color(GREEN)
        
        # Resultant wave: shift between 0 and PI/2
        result_phase_shift = 0.6 
        
        def get_green_wave():
            return FunctionGraph(
                lambda x: 0.9 * WAVE_AMPLITUDE * np.sin(2 * PI * (x - time_tracker.get_value()) - result_phase_shift),
                x_range=WAVE_X_RANGE,
                color=GREEN,
                stroke_width=6
            )
        
        green_wave = get_green_wave()
        green_wave.add_updater(lambda m: m.become(get_green_wave()))
        green_wave.move_to(white_wave.get_center())

        self.play(
            white_wave.animate.set_stroke(opacity=0.3),
            cyan_wave.animate.set_stroke(opacity=0.3),
            FadeIn(green_wave),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # Display text "Phase Lag -> Higher Refractive Index" (#FFD700) at the bottom.
        self.lecture[4].set_color("#FFD700")
        
        conclusion_text = Text("Phase Lag -> Higher Refractive Index", font_size=24, color="#FFD700")
        # Issue 34: Use place_in_area F2-F5
        self.place_in_area(conclusion_text, 'F2', 'F5', scale_factor=0.9)
        
        self.play(Write(conclusion_text))
        self.wait(3)
