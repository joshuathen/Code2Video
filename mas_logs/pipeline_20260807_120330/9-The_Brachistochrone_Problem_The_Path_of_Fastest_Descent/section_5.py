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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup the scene
        title = "The Solution: The Cycloid"
        lines = [
            "This optimal path is called a cycloid.",
            "A cycloid is traced by a point on a rolling wheel.",
            "It perfectly balances high speed and short distance.",
            "Mathematics proves the cycloid is the fastest."
        ]
        self.setup_layout(title, lines)

        # Colors
        color_path = "#00FFFF"
        color_wheel = "#CCCCCC"
        color_dot = "#FF0000"
        color_label = "#FFFF00"

        # Assets
        wheel_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/wheel.svg"

        # === Animation for Lecture Line 1 ===
        # "This optimal path is called a cycloid."
        self.play(self.lecture[0].animate.set_color(color_label), run_time=0.5)
        
        # Fix for Issue 32: Move label from A3 to B4
        label = Text("Cycloid", color=color_label)
        self.place_at_grid(label, "B4", scale_factor=1.0)
        self.play(Write(label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "A cycloid is traced by a point on a rolling wheel."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color_wheel),
            run_time=0.5
        )

        # Fix for Issue 21: Use SVGMobject for the wheel [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/wheel.svg]
        # Wheel parameters
        radius = 0.4
        # Floor is at F2-F6
        # Start at E2 to avoid overlapping with lecture text (Belief B021)
        start_center = self.grid["E2"] + UP * radius
        
        wheel = SVGMobject(wheel_asset_path).set_color(color_wheel)
        wheel.height = 2 * radius
        wheel.move_to(start_center)
        
        dot = Dot(color=color_dot).scale(1.2)
        # Position dot at the bottom of the wheel initially (t=0)
        dot.move_to(wheel.get_bottom())

        wheel_group = VGroup(wheel, dot)
        
        # Line for the ground - using F2 to F6 to respect margins
        ground = Line(self.grid["F2"], self.grid["F6"], color=GREY_D)
        
        self.play(Create(ground), FadeIn(wheel), FadeIn(dot))
        
        # Tracer for the cycloid path (#00FFFF)
        path_tracker = TracedPath(dot.get_center, stroke_color=color_path, stroke_width=4)
        self.add(path_tracker)

        # Roller animation [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/wheel.svg]
        roll_tracker = ValueTracker(0)
        wheel.last_t = 0

        def update_wheel(mob):
            t = roll_tracker.get_value()
            dt = t - mob.last_t
            mob.rotate(-dt)
            mob.move_to(start_center + RIGHT * (radius * t))
            mob.last_t = t

        def update_dot(mob):
            t = roll_tracker.get_value()
            dist = radius * t
            center = start_center + RIGHT * dist
            # Dot position relative to center: (-r*sin(t), -r*cos(t))
            offset = np.array([-radius * np.sin(t), -radius * np.cos(t), 0])
            mob.move_to(center + offset)

        wheel.add_updater(update_wheel)
        dot.add_updater(update_dot)

        # Calculate max t to reach col 6
        total_dist = (self.grid["E6"][0] - self.grid["E2"][0])
        max_t = total_dist / radius

        self.play(roll_tracker.animate.set_value(max_t), run_time=5, rate_func=linear)
        
        wheel.remove_updater(update_wheel)
        dot.remove_updater(update_dot)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "It perfectly balances high speed and short distance."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color_path),
            run_time=0.5
        )
        
        # Highlight path
        self.play(path_tracker.animate.set_stroke(width=8), run_time=0.5)
        self.play(path_tracker.animate.set_stroke(width=4), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Mathematics proves the cycloid is the fastest."
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(GOLD),
            run_time=0.5
        )
        
        # Focus on final result
        self.play(
            wheel.animate.set_opacity(0.3),
            dot.animate.set_opacity(0.3),
            label.animate.scale(1.2).set_color(GOLD),
            run_time=1
        )

        self.wait(2)

        # Cleanup
        self.play(
            FadeOut(label),
            FadeOut(wheel_group),
            FadeOut(path_tracker),
            FadeOut(ground),
            self.lecture[3].animate.set_color(WHITE)
        )
