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
        # Setup Layout
        title = "Wave Superposition and the Phase Delay"
        lines = [
            "Moving charges re-radiate their own secondary waves.",
            "Electron mass causes a delay in this response.",
            "The total field is the sum of all waves.",
            "Superposition creates a new wave with a phase shift.",
            "This phase delay slows the collective wavefront."
        ]
        self.setup_layout(title, lines)

        # Assets & Objects
        electron_asset_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/electron.svg"
        electron = SVGMobject(electron_asset_path)
        self.place_at_grid(electron, "B3", scale_factor=0.6)
        electron.set_color(BLUE_A)

        time_tracker = ValueTracker(0)
        
        # Secondary Waves (Concentric Circles)
        secondary_waves = VGroup()
        def create_expanding_circle():
            circle = Circle(radius=0.1, color="#00FFFF", stroke_opacity=0.8)
            circle.move_to(electron.get_center())
            return circle

        # Incoming Wave and Electron Phase tracking
        # We'll use a simple sine function for visualization
        amplitude = 0.5
        frequency = 1.5
        phase_lag = PI / 4

        # Tracking lines
        incoming_peak_line = Line(UP, DOWN, color=GREY_B).scale(0.5)
        electron_peak_line = Line(UP, DOWN, color="#FFD700").scale(0.5)
        phase_lag_label = Text("Phase Lag", font_size=18, color="#FFD700")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(TEAL)
        self.add(electron)
        
        # Electron oscillation updater
        electron.add_updater(lambda m: m.move_to(self.grid["B3"] + UP * amplitude * np.sin(time_tracker.get_value() * frequency * 2)))
        
        # Secondary waves emission (simulated)
        last_emit_time = 0
        def update_secondary_waves(group, dt):
            nonlocal last_emit_time
            curr_time = time_tracker.get_value()
            if curr_time - last_emit_time > 0.8:
                new_c = create_expanding_circle()
                group.add(new_c)
                last_emit_time = curr_time
            
            for c in group:
                c.scale(1 + 2 * dt)
                c.set_stroke(opacity=max(0, c.get_stroke_opacity() - 0.5 * dt))
                if c.get_stroke_opacity() <= 0:
                    group.remove(c)

        secondary_waves.add_updater(update_secondary_waves)
        self.add(secondary_waves)
        
        self.play(time_tracker.animate.set_value(3), run_time=3, rate_func=linear)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Position tracking lines relative to the oscillation
        incoming_peak_line.add_updater(lambda m: m.move_to(self.grid["B3"] + RIGHT * 1.0 + UP * amplitude * np.sin(time_tracker.get_value() * frequency * 2 + phase_lag)))
        electron_peak_line.add_updater(lambda m: m.move_to(self.grid["B3"] + RIGHT * 1.3 + UP * amplitude * np.sin(time_tracker.get_value() * frequency * 2)))
        
        self.place_at_grid(phase_lag_label, "B4", scale_factor=0.8)
        
        self.play(FadeIn(incoming_peak_line), FadeIn(electron_peak_line), FadeIn(phase_lag_label))
        self.play(time_tracker.animate.set_value(6), run_time=3, rate_func=linear)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(BLUE_C)

        # Clear upper clutter for wave comparison
        self.play(FadeOut(incoming_peak_line), FadeOut(electron_peak_line), FadeOut(phase_lag_label), secondary_waves.animate.set_stroke(opacity=0))
        secondary_waves.remove_updater(update_secondary_waves)

        # Define wave area
        wave_x_start = self.grid["D1"][0]
        wave_x_end = self.grid["D6"][0]
        wave_width = wave_x_end - wave_x_start

        def get_wave_points(offset_y, phase_shift=0, amp=0.4, color=WHITE):
            pts = []
            for dx in np.linspace(0, wave_width, 50):
                x = wave_x_start + dx
                y = offset_y + amp * np.sin(3 * dx - time_tracker.get_value() * 4 + phase_shift)
                pts.append([x, y, 0])
            return pts

        primary_wave = VMobject(color=GREY_A)
        secondary_wave_viz = VMobject(color="#00FFFF")

        primary_wave.add_updater(lambda m: m.set_points_as_corners(get_wave_points(self.grid["D3"][1])))
        secondary_wave_viz.add_updater(lambda m: m.set_points_as_corners(get_wave_points(self.grid["E3"][1], phase_shift=-phase_lag, amp=0.2)))

        self.add(primary_wave, secondary_wave_viz)
        self.play(time_tracker.animate.set_value(9), run_time=3, rate_func=linear)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(WHITE) # Already white, set to highlight? Prompt says no color animations except for lines.
        self.lecture[3].set_color(YELLOW_A)

        resultant_wave = VMobject(color=WHITE, stroke_width=4)
        
        def get_resultant_points(offset_y):
            pts = []
            for dx in np.linspace(0, wave_width, 50):
                x = wave_x_start + dx
                # Sum of Primary (amp 0.4) and Secondary (amp 0.2, phase lag)
                y_val = 0.4 * np.sin(3 * dx - time_tracker.get_value() * 4) + 0.2 * np.sin(3 * dx - time_tracker.get_value() * 4 - phase_lag)
                pts.append([x, offset_y + y_val, 0])
            return pts

        resultant_wave.add_updater(lambda m: m.set_points_as_corners(get_resultant_points(self.grid["F3"][1])))
        
        self.play(FadeIn(resultant_wave))
        self.play(time_tracker.animate.set_value(12), run_time=3, rate_func=linear)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(GOLD)

        # Wavefronts comparison
        # Vertical bars tracking peaks of original vs resultant
        wavefront_vacuum = always_redraw(lambda: VGroup(*[
            Line(UP, DOWN, color=GREY_E, stroke_opacity=0.5).scale(0.3).move_to([wave_x_start + ( (time_tracker.get_value()*4/3 + i*2*PI/3) % wave_width ), self.grid["F3"][1] + 1.0, 0])
            for i in range(3)
        ]))
        
        wavefront_resultant = always_redraw(lambda: VGroup(*[
            Line(UP, DOWN, color=WHITE).scale(0.3).move_to([wave_x_start + ( ( (time_tracker.get_value()*4 - 0.2)/3 + i*2*PI/3 ) % wave_width ), self.grid["F3"][1], 0])
            for i in range(3)
        ]))

        self.add(wavefront_vacuum, wavefront_resultant)
        
        # Slow down simulation to see the spacing/lag
        self.play(time_tracker.animate.set_value(15), run_time=3, rate_func=linear)
        
        # Cleanup
        self.wait(2)
