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
        # Data
        title = "Prerequisite: The Chain Rule Intuition"
        lecture_lines = [
            "Changes in early layers ripple through to the output.",
            "The Chain Rule calculates this chain of influence mathematically.",
            "It links how weights affect the final error score."
        ]
        
        # Colors
        COLOR_NEUTRAL = "#C0C0C0"
        COLOR_GEARS = "#FFFF00"
        COLOR_ERROR = "#FF00FF"
        
        # Assets
        GEAR_PATH = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/gears.svg"
        
        # Setup
        self.setup_layout(title, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Show three connected gears [Asset: gears.svg] labeled 'Weight A', 'Neuron B', and 'Error'. Color: #C0C0C0.
        self.play(self.lecture[0].set_color(COLOR_NEUTRAL))
        
        gear_a = SVGMobject(GEAR_PATH).set_color(COLOR_NEUTRAL)
        self.place_at_grid(gear_a, "B2", scale_factor=0.6)
        label_a = Text("Weight A", font_size=18, color=COLOR_NEUTRAL)
        label_a.next_to(gear_a, DOWN, buff=0.2)
        
        gear_b = SVGMobject(GEAR_PATH).set_color(COLOR_NEUTRAL)
        self.place_at_grid(gear_b, "C3", scale_factor=0.6)
        label_b = Text("Neuron B", font_size=18, color=COLOR_NEUTRAL)
        label_b.next_to(gear_b, DOWN, buff=0.2)
        
        gear_c = SVGMobject(GEAR_PATH).set_color(COLOR_NEUTRAL)
        self.place_at_grid(gear_c, "D4", scale_factor=0.6)
        label_c = Text("Error", font_size=18, color=COLOR_NEUTRAL)
        label_c.next_to(gear_c, DOWN, buff=0.2)
        
        gears_group = VGroup(gear_a, label_a, gear_b, label_b, gear_c, label_c)
        self.play(FadeIn(gears_group), run_time=1.5)
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Animate Gear A rotating, which mechanically causes Gear B to turn. Color: #FFFF00.
        self.play(
            self.lecture[0].set_color(WHITE),
            self.lecture[1].set_color(COLOR_GEARS),
            gear_a.animate.set_color(COLOR_GEARS),
            gear_b.animate.set_color(COLOR_GEARS),
            label_a.animate.set_color(COLOR_GEARS),
            label_b.animate.set_color(COLOR_GEARS)
        )
        
        # Use ValueTracker for rotation state
        rot_tracker = ValueTracker(0)
        
        # Define internal state for absolute rotation calculation to avoid drift
        gear_a.last_rot = 0.0
        gear_b.last_rot = 0.0
        gear_c.last_rot = 0.0

        def update_rotating_gear(m, tracker, direction):
            current_val = tracker.get_value()
            delta = current_val - m.last_rot
            m.rotate(direction * delta * DEGREES, about_point=m.get_center())
            m.last_rot = current_val

        # Add updaters for gears
        gear_a.add_updater(lambda m: update_rotating_gear(m, rot_tracker, direction=-1))
        gear_b.add_updater(lambda m: update_rotating_gear(m, rot_tracker, direction=1))
        gear_c.add_updater(lambda m: update_rotating_gear(m, rot_tracker, direction=-1))
        
        # Animate rotation
        self.play(rot_tracker.animate.set_value(360.0), run_time=4, rate_func=linear)
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Gear B turning causes the 'Error' indicator to move on a scale. Color: #FF00FF.
        self.play(
            self.lecture[1].set_color(WHITE),
            self.lecture[2].set_color(COLOR_ERROR),
            gear_c.animate.set_color(COLOR_ERROR),
            label_c.animate.set_color(COLOR_ERROR)
        )
        
        # Create Error Indicator Scale
        scale_arc = Arc(radius=0.6, start_angle=0, angle=PI, color=COLOR_ERROR)
        self.place_at_grid(scale_arc, "D5", scale_factor=1.0)
        scale_arc.shift(RIGHT * 0.5) # Fine adjustment for visual balance
        
        # Create Needle
        needle_center = scale_arc.get_center()
        needle = Line(needle_center, needle_center + UP * 0.5, color=COLOR_ERROR, stroke_width=5)
        
        def update_needle(m):
            # Oscillation linked to the rotation tracker
            oscillation_angle = np.sin(rot_tracker.get_value() * DEGREES * 0.5) * 45.0
            # Reset needle geometry based on oscillation
            target_vector = np.array([
                -np.sin(oscillation_angle * DEGREES),
                np.cos(oscillation_angle * DEGREES),
                0
            ]) * 0.5
            m.put_start_and_end_on(needle_center, needle_center + target_vector)

        needle.add_updater(update_needle)
        self.add(scale_arc, needle)
        
        # Continue rotation to show error linkage
        self.play(rot_tracker.animate.set_value(1080.0), run_time=6, rate_func=linear)
        self.wait(2)
        
        # Cleanup updaters
        gear_a.clear_updaters()
        gear_b.clear_updaters()
        gear_c.clear_updaters()
        needle.clear_updaters()
