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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup layout
        title_text = "The Winding Machine (The Core Mechanism)"
        lecture_lines = [
            "We wrap our complex signal around a central point.",
            "The wrapping speed depends on a specific \"winding frequency.\"",
            "Most winding frequencies result in a balanced, circular mess.",
            "But certain frequencies create a distinct, lopsided shape.",
            "This happens when the winding matches a hidden component."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        WIRE_COLOR = "#ECF0F1"
        MATCH_COLOR = "#F1C40F"

        # Signal function: 1.4 + 0.6*cos(2*pi*2*t)
        # This signal has a component at 2.0 Hz.
        def signal_func(t):
            return 1.4 + 0.6 * np.cos(2 * np.pi * 2.0 * t)

        def get_wrapped_points(f, t_max=4.0, step=0.01):
            pts = []
            for t in np.arange(0, t_max + step, step):
                r = signal_func(t)
                # Clockwise winding is the convention for Fourier transforms (e^-iwt)
                theta = -2 * np.pi * f * t
                pts.append([r * np.cos(theta), r * np.sin(theta), 0])
            return pts

        # Area center for wrapping machine (using B1 to F6 as per Issue 53 fix)
        tl_pos = self.grid["B1"]
        br_pos = self.grid["F6"]
        area_center = (tl_pos + br_pos) / 2
        # Radius of signal is approx 2.0. Area height/width is ~4.0.
        # Scale factor from fix is 0.9.
        SCALE = 0.55 # Adjusted to fit the 0.9 area constraint nicely

        # === Animation for Lecture Line 1 ===
        # Represent the complex white signal as a horizontal wire (#ECF0F1).
        self.lecture[0].set_color(WIRE_COLOR)
        
        # Initial wire: horizontal representation of the signal values
        wire_pts = [ [ (t - 2.0) * 0.8, (signal_func(t) - 1.4) * 1.5, 0] for t in np.arange(0, 4.01, 0.05) ]
        wire = VMobject(color=WIRE_COLOR)
        wire.set_points_as_corners(wire_pts)
        # Apply fix from Issue 53: place in B1-F6 with scale 0.9
        self.place_in_area(wire, "B1", "F6", scale_factor=0.9)
        
        self.play(Create(wire))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Wrap the wire around a central point to form a spiral.
        self.lecture[1].set_color(WIRE_COLOR)
        
        f_tracker = ValueTracker(0.1)
        
        # Target wrapped wire starting shape
        wrapped_pts_start = get_wrapped_points(f_tracker.get_value())
        scaled_wrapped_pts_start = [np.array(p) * SCALE + area_center for p in wrapped_pts_start]
        
        wrapped_wire = VMobject(color=WIRE_COLOR)
        wrapped_wire.set_points_as_corners(scaled_wrapped_pts_start)
        
        self.play(ReplacementTransform(wire, wrapped_wire))
        
        # Updater for winding frequency based on tracker
        def wire_updater(mob):
            f = f_tracker.get_value()
            pts = get_wrapped_points(f)
            # Apply fixed scaling and center shift
            new_pts = [np.array(p) * SCALE + area_center for p in pts]
            mob.set_points_as_corners(new_pts)

        wrapped_wire.add_updater(wire_updater)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Increase the wrapping speed, showing the spiral changing shape.
        self.lecture[2].set_color(WIRE_COLOR)
        
        # Sweep through non-matching frequencies (balanced/circular mess)
        self.play(f_tracker.animate.set_value(0.7), run_time=2, rate_func=linear)
        self.play(f_tracker.animate.set_value(1.4), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # At a specific speed, the spiral becomes heavily offset to one side.
        self.lecture[3].set_color(WIRE_COLOR)
        
        # Move towards the matching frequency (2.0 Hz)
        self.play(f_tracker.animate.set_value(2.0), run_time=3)
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # The lopsided spiral flashes Yellow (#F1C40F) to indicate a match.
        self.lecture[4].set_color(MATCH_COLOR)
        
        # Remove updater to perform discrete flash animation and avoid interference
        wrapped_wire.remove_updater(wire_updater)
        
        # Successive flash pulses
        for _ in range(2):
            self.play(
                wrapped_wire.animate.set_color(MATCH_COLOR).set_stroke(width=8),
                run_time=0.4
            )
            self.play(
                wrapped_wire.animate.set_color(WIRE_COLOR).set_stroke(width=4),
                run_time=0.4
            )
        
        # Final color set to Yellow
        self.play(wrapped_wire.animate.set_color(MATCH_COLOR), run_time=0.3)
        self.wait(2)
