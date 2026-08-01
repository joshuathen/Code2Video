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

class Section1Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title_text = "The Hook: Turbo the Cheetah’s Great Run"
        lines_text = [
            "Meet Turbo the Cheetah, sprinting across the savanna.",
            "If we know his speed, can we find distance?",
            "If we know distance, can we find speed?"
        ]
        
        self.setup_layout(title_text, lines_text)
        
        # Colors
        TURBO_COLOR = "#FFD700"
        SAVANNA_COLOR = "#228B22"
        VELOCITY_COLOR = "#00BFFF"
        DISPLACEMENT_COLOR = "#FFFF00"
        MAP_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Savanna floor
        floor = Line(self.grid["F1"] + LEFT*0.5, self.grid["F6"] + RIGHT*0.5, color=SAVANNA_COLOR, stroke_width=6)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/che.svg]
        # Turbo silhouette using the provided SVG asset
        turbo_silhouette = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/che.svg")
        turbo_silhouette.set_color(TURBO_COLOR)
        # Apply scale factor 1.1 as per Issue 31
        self.place_at_grid(turbo_silhouette, "F1", scale_factor=1.1)
        # Adjust position slightly above floor
        turbo_silhouette.shift(UP * 0.4)
        
        turbo_label = Text("Turbo", font_size=18, color=TURBO_COLOR).next_to(turbo_silhouette, UP, buff=0.1)
        turbo_group = VGroup(turbo_silhouette, turbo_label)

        # Speedometer setup
        speed_arc = Arc(start_angle=PI, angle=-PI, radius=0.7, color=WHITE)
        speed_label = Text("Velocity", font_size=18, color=WHITE).next_to(speed_arc, UP, buff=0.2)
        speed_needle = Line(speed_arc.get_center(), speed_arc.get_center() + LEFT * 0.6, color=RED)
        speedometer = VGroup(speed_arc, speed_label, speed_needle)
        
        # Placement fix as per Issue 30: move to B4-C6 with scale 0.8
        self.place_in_area(speedometer, "B4", "C6", scale_factor=0.8)
        
        # Map Tracker setup
        map_line_base = Line(self.grid["E1"], self.grid["E6"], color=GREY, stroke_width=2)
        # Initializing map_tracker_line as a zero-length line at the start
        map_tracker_line = Line(self.grid["E1"], self.grid["E1"], color=MAP_COLOR, stroke_width=4)
        map_label = Text("Displacement", font_size=18, color=WHITE).next_to(map_line_base, UP, buff=0.1)
        
        # Grouping tracker elements
        map_group = VGroup(map_line_base, map_tracker_line, map_label)

        # Initial appearance of elements
        self.play(
            Create(floor),
            FadeIn(turbo_group),
            Create(speed_arc),
            Write(speed_label),
            Create(speed_needle),
            Create(map_line_base),
            Write(map_label)
        )
        
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color(TURBO_COLOR))
        
        # Use ValueTracker for coordinating the movement and instrumentation
        progress = ValueTracker(0)
        
        # Add updaters for smooth persistent animation
        # Moving Turbo across the grid from F1 to F6
        turbo_group.add_updater(lambda m: m.move_to(
            self.grid["F1"] + progress.get_value() * (self.grid["F6"] - self.grid["F1"]) + UP * 0.4
        ))
        
        # Updating speedometer needle based on progress
        speed_needle.add_updater(lambda m: m.set_points_by_ends(
            speed_arc.get_center(),
            speed_arc.get_center() + np.array([
                np.cos(PI - progress.get_value() * PI * 0.8), 
                np.sin(PI - progress.get_value() * PI * 0.8), 
                0
            ]) * 0.6 * 0.8 # needle length adjusted relative to speedometer scaling
        ))
        
        # Updating the map tracker line (distance traveled)
        map_tracker_line.add_updater(lambda m: m.set_points_by_ends(
            self.grid["E1"],
            self.grid["E1"] + progress.get_value() * (self.grid["E6"] - self.grid["E1"])
        ))

        self.add(turbo_group, speed_needle, map_tracker_line)
        self.play(progress.animate.set_value(1), run_time=5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "If we know his speed, can we find distance?"
        self.play(self.lecture[1].animate.set_color(VELOCITY_COLOR))
        self.play(
            speed_arc.animate.set_color(VELOCITY_COLOR),
            speed_label.animate.set_color(VELOCITY_COLOR),
            speed_needle.animate.set_color(VELOCITY_COLOR),
            Flash(speed_arc, color=VELOCITY_COLOR)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "If we know distance, can we find speed?"
        self.play(self.lecture[2].animate.set_color(DISPLACEMENT_COLOR))
        self.play(
            map_tracker_line.animate.set_color(DISPLACEMENT_COLOR),
            map_line_base.animate.set_color(DISPLACEMENT_COLOR),
            map_label.animate.set_color(DISPLACEMENT_COLOR),
            Flash(map_tracker_line, color=DISPLACEMENT_COLOR)
        )
        
        # Cleanup updaters to prevent unexpected behavior in future frames
        turbo_group.clear_updaters()
        speed_needle.clear_updaters()
        map_tracker_line.clear_updaters()
        
        self.wait(2)
