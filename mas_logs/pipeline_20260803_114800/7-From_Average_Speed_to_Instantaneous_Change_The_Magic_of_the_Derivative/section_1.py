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
        # Initial Setup
        title = "The Speedometer Paradox"
        lines = [
            "Calculating average speed needs distance over time.",
            "But how fast is a cheetah *exactly* now?",
            "The speedometer shows speed at a single instant."
        ]
        self.setup_layout(title, lines)

        # Pre-load Assets
        cheetah_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        speed_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speed.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Speedometer setup
        speed_tracker = ValueTracker(0)
        
        # Speedometer components
        speedometer_base = speed_svg.copy()
        speed_num = DecimalNumber(0, num_decimal_places=0, font_size=36, color=WHITE)
        unit_label = Text("km/h", font_size=18, color=GREY_B)
        needle = Line(ORIGIN, LEFT * 0.7, color=RED, stroke_width=4)
        
        speedometer_group = VGroup(speedometer_base, speed_num, unit_label, needle)
        self.place_in_area(speedometer_group, "D4", "F6", scale_factor=1.1)
        
        # Relative positioning inside the group
        gauge_center = speedometer_base.get_center()
        speed_num.move_to(gauge_center + DOWN * 0.4)
        unit_label.next_to(speed_num, DOWN, buff=0.1)
        needle.shift(gauge_center)
        
        # Updaters for speedometer
        speed_num.add_updater(lambda d: d.set_value(speed_tracker.get_value()))
        
        def update_needle(m):
            val = speed_tracker.get_value()
            # Map 0-100 to PI to 0 (semi-circle)
            angle = PI - (val / 100) * PI
            length = 0.7 * 1.1 
            new_end = gauge_center + np.array([np.cos(angle), np.sin(angle), 0]) * length
            m.set_points_as_corners([gauge_center, new_end])
        
        needle.add_updater(update_needle)
        
        # Cheetah setup (Starting at B2 per Issue 23)
        cheetah = cheetah_svg.copy()
        self.place_at_grid(cheetah, "B2", scale_factor=0.8)
        cheetah_label = Text("Cheetah", font_size=18, color=WHITE).next_to(cheetah, UP, buff=0.1)
        cheetah_full = VGroup(cheetah, cheetah_label)
        
        # Path from B2 to B6
        path = Line(self.grid["B2"], self.grid["B6"], color=GREY_E, stroke_width=2)
        dist_label = Text("Distance", font_size=18, color=GREY_C).next_to(path, UP, buff=0.8)
        
        self.add(path, speedometer_group)
        self.play(FadeIn(cheetah_full, dist_label))
        
        # Movement 1
        self.play(
            cheetah_full.animate.move_to(self.grid["B6"]),
            speed_tracker.animate.set_value(100),
            run_time=4,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Reset position to B2
        self.play(
            cheetah_full.animate.move_to(self.grid["B2"]),
            speed_tracker.animate.set_value(0),
            run_time=1
        )
        
        # Freeze at instant (visualized at 75% of path)
        freeze_pos = self.grid["B2"] + 0.75 * (self.grid["B6"] - self.grid["B2"])
        
        self.play(
            cheetah_full.animate.move_to(freeze_pos),
            speed_tracker.animate.set_value(100),
            run_time=2.5,
            rate_func=linear
        )
        
        # Visual highlight of the 'moment'
        self.play(Indicate(cheetah_full, color=YELLOW, scale_factor=1.1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Highlight speed value
        glow_rect = SurroundingRectangle(speed_num, color="#FFFF00", buff=0.1)
        
        self.play(
            speed_num.animate.set_color("#FFFF00"),
            Create(glow_rect)
        )
        
        # Paradox Text (Per Issue 22: C1 to C3)
        paradox_text = Text("Distance/Time = 0/0?", color="#FF0000", font_size=28)
        self.place_in_area(paradox_text, "C1", "C3", scale_factor=0.8)
        
        self.play(Write(paradox_text))
        self.play(Flash(paradox_text, color="#FF0000"))
        
        self.wait(3)
