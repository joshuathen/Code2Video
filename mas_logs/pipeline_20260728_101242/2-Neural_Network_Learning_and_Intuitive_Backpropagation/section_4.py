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
        title = "Intuitive Backpropagation: The Blame Game"
        lines = [
            "Backpropagation identifies which weights caused the error.",
            "We work backward from the output to input.",
            "Think of tracing blame through a relay race.",
            "The final error tells us what to change.",
            "We distribute adjustments based on individual contributions."
        ]
        self.setup_layout(title, lines)

        # Assets
        GEAR_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/gears.svg"
        RELAY_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/relay.svg"

        # Initialize velocity tracker for updaters
        self.gear_vel = ValueTracker(2.0)

        # === Animation for Lecture Line 1 ===
        # Show three connected gray gears ([Asset: gears.svg]) spinning; 
        # final gear reaches a red 'Error' marker (#FF0000).
        self.lecture[0].set_color("#A9A9A9")
        
        gear1 = SVGMobject(GEAR_ASSET).set_color("#A9A9A9")
        gear2 = SVGMobject(GEAR_ASSET).set_color("#A9A9A9")
        gear3 = SVGMobject(GEAR_ASSET).set_color("#A9A9A9")

        # Issue 37: Reposition gears to 'C4', 'C5', 'C6', with scale_factor 0.7
        self.place_at_grid(gear1, "C4", scale_factor=0.7)
        self.place_at_grid(gear2, "C5", scale_factor=0.7)
        self.place_at_grid(gear3, "C6", scale_factor=0.7)

        # Add persistent rotation updaters
        gear1.add_updater(lambda g, dt: g.rotate(self.gear_vel.get_value() * dt))
        gear2.add_updater(lambda g, dt: g.rotate(-self.gear_vel.get_value() * dt))
        gear3.add_updater(lambda g, dt: g.rotate(self.gear_vel.get_value() * dt))

        self.add(gear1, gear2, gear3)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # We work backward from the output to input.
        # A red 'Blame' pulse (#FF0000) travels backward from output gear to input gear.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF0000")

        # Issue 38: error_marker 'D6', error_label 'D5', scale_factor 0.8
        error_marker = Star(color="#FF0000", fill_opacity=1).scale(0.2)
        self.place_at_grid(error_marker, "D6", scale_factor=0.8)
        error_label = Text("Error", color="#FF0000", font_size=16)
        self.place_at_grid(error_label, "D5", scale_factor=0.8)

        self.play(FadeIn(error_marker), FadeIn(error_label))

        # Red pulse from output (C6) to input (C4)
        # Issue 39: blame_pulse starts at 'C6'
        blame_pulse = Circle(radius=0.1, color="#FF0000", stroke_width=4)
        self.place_at_grid(blame_pulse, "C6", scale_factor=1.0)
        
        self.add(blame_pulse)
        self.play(
            blame_pulse.animate.move_to(self.grid["C4"]).set_stroke(width=10).scale(1.5),
            run_time=2,
            rate_func=slow_into
        )
        self.play(FadeOut(blame_pulse))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Think of tracing blame through a relay race.
        # A relay race icon ([Asset: relay.svg]) traces the blame path through the gears.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(WHITE)

        relay_icon = SVGMobject(RELAY_ASSET).set_color(WHITE)
        self.place_at_grid(relay_icon, "C6", scale_factor=0.5)

        self.play(FadeIn(relay_icon))
        self.play(relay_icon.animate.move_to(self.grid["C5"]), run_time=0.8)
        self.play(relay_icon.animate.move_to(self.grid["C4"]), run_time=0.8)
        self.play(FadeOut(relay_icon))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The final error tells us what to change.
        # Display the final error value (#FF0000) indicating the necessary total change.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FF0000")

        error_value = Text("Error Score: 0.82", color="#FF0000", font_size=20)
        self.place_at_grid(error_value, "E6", scale_factor=1.0)
        self.play(Write(error_value))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # We distribute adjustments based on individual contributions.
        # Gears glow red (#FF0000) proportionally to their contribution and adjust rotation speed.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FF0000")

        # Adjust rotation speed and color
        self.play(
            gear3.animate.set_color("#FF0000"),
            gear2.animate.set_color("#CC0000"),
            gear1.animate.set_color("#880000"),
            self.gear_vel.animate.set_value(5.0),
            run_time=3
        )
        
        # Cleanup
        self.play(
            gear1.animate.set_color("#A9A9A9"),
            gear2.animate.set_color("#A9A9A9"),
            gear3.animate.set_color("#A9A9A9"),
            FadeOut(error_marker),
            FadeOut(error_label),
            FadeOut(error_value),
            self.gear_vel.animate.set_value(2.0),
            run_time=1
        )
        
        self.wait(1)
        self.lecture[4].set_color(WHITE)
