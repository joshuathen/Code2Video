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

class Section2Scene(TeachingScene):
    def construct(self):
        # --- Data Setup ---
        title_text = "Prerequisite: The Language of Waves"
        lecture_lines = [
            "Light waves have both amplitude and phase properties.",
            "Interference happens when two light waves overlap.",
            "Aligned peaks create brightness, while opposing peaks cancel out."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_WAVE = "#FFFFFF"
        COLOR_PEAK = "#FFFF00"
        COLOR_TROUGH = "#FF00FF"
        COLOR_INTERFERENCE = "#00FFFF"
        COLOR_DIM = "#666666"

        # Determine area bounds for A3 to F6 to avoid overlapping lecture text
        tl = self.grid["A3"]
        br = self.grid["F6"]
        area_center = (tl + br) / 2
        
        start_x = tl[0]
        end_x = br[0]
        wave_y = area_center[1]
        wave_width = end_x - start_x

        # Trackers for animation
        phase_tracker = ValueTracker(0.0)
        amp_tracker = ValueTracker(0.7)
        phase_tracker2 = ValueTracker(np.pi) # Second wave offset for initial interference

        def get_wave_points(phase, amp):
            res = 80
            pts = []
            for i in range(res + 1):
                pct = float(i) / res
                curr_x = start_x + pct * wave_width
                # Wave equation: y = baseline + amp * sin(2*pi*cycles*pct - phase)
                # 2 cycles across the width
                curr_y = wave_y + float(amp) * np.sin(4.0 * np.pi * pct - float(phase))
                pts.append([curr_x, curr_y, 0.0])
            return pts

        # === Animation for Lecture Line 1 ===
        # A white sine wave #FFFFFF oscillates horizontally.
        self.play(self.lecture[0].animate.set_color(COLOR_WAVE), run_time=0.5)
        
        # Persistent wave mobject with updater
        wave1 = VMobject(color=COLOR_WAVE)
        # Positioning: Explicitly restricted to area A3-F6 via start_x and wave_y
        wave1.set_points_as_corners(get_wave_points(0.0, 0.7))
        wave1.add_updater(lambda m: m.set_points_as_corners(
            get_wave_points(phase_tracker.get_value(), amp_tracker.get_value())
        ))
        
        self.add(wave1)
        self.play(phase_tracker.animate.set_value(2.0 * np.pi), run_time=3, rate_func=linear)

        # === Animation for Lecture Line 2 ===
        # The peak of the wave flashes yellow #FFFF00 and the trough flashes magenta #FF00FF.
        self.play(
            self.lecture[0].animate.set_color(COLOR_DIM),
            self.lecture[1].animate.set_color(WHITE),
            run_time=0.5
        )

        peak_dot = Dot(color=COLOR_PEAK, radius=0.1)
        trough_dot = Dot(color=COLOR_TROUGH, radius=0.1)

        def update_peak(d):
            p = float(phase_tracker.get_value())
            amp = float(amp_tracker.get_value())
            # Find pct such that sin(4*pi*pct - p) = 1
            # 4*pi*pct - p = pi/2 + 2*k*pi
            pct = (np.pi / 2.0 + p) / (4.0 * np.pi)
            pct = pct % 1.0 # Cycle through both peaks if necessary
            d.move_to([start_x + pct * wave_width, wave_y + amp, 0.0])

        def update_trough(d):
            p = float(phase_tracker.get_value())
            amp = float(amp_tracker.get_value())
            # Find pct such that sin(4*pi*pct - p) = -1
            # 4*pi*pct - p = 3*pi/2 + 2*k*pi
            pct = (1.5 * np.pi + p) / (4.0 * np.pi)
            pct = pct % 1.0
            d.move_to([start_x + pct * wave_width, wave_y - amp, 0.0])

        peak_dot.add_updater(update_peak)
        trough_dot.add_updater(update_trough)
        
        # Positioning: Dots grouped and constrained to the same area as the wave
        dots_group = VGroup(peak_dot, trough_dot)
        self.add(dots_group)
        
        self.play(phase_tracker.animate.set_value(4.0 * np.pi), run_time=3, rate_func=linear)
        self.remove(dots_group)

        # === Animation for Lecture Line 3 ===
        # Two waves align their peaks, merging into a single wave with double the amplitude in cyan #00FFFF.
        self.play(
            self.lecture[1].animate.set_color(COLOR_DIM),
            self.lecture[2].animate.set_color(COLOR_INTERFERENCE),
            run_time=0.5
        )

        # Introduce secondary wave (initially out of phase)
        # Positioning: Also restricted to A3-F6
        wave2 = VMobject(color=WHITE, stroke_opacity=0.4)
        wave2.set_points_as_corners(get_wave_points(phase_tracker2.get_value(), 0.7))
        wave2.add_updater(lambda m: m.set_points_as_corners(
            get_wave_points(phase_tracker2.get_value(), 0.7)
        ))
        
        self.add(wave2)
        self.wait(1.0)

        # Phase alignment (Transition to constructive interference)
        self.play(
            phase_tracker2.animate.set_value(phase_tracker.get_value()),
            run_time=2.0
        )
        self.wait(0.5)

        # Merge: increase amplitude of first wave and fade out the second
        self.play(
            FadeOut(wave2),
            amp_tracker.animate.set_value(1.4),
            wave1.animate.set_color(COLOR_INTERFERENCE),
            run_time=2.0
        )
        
        # Final oscillation of the constructive interference result
        self.play(phase_tracker.animate.set_value(6.0 * np.pi), run_time=3, rate_func=linear)
        self.wait(2.0)
