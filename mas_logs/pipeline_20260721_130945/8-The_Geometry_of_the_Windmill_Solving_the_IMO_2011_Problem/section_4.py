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
        # 1. Setup Layout
        title_text = "The Invariant Property: The Balancing Act"
        lecture_lines = [
            "Watch the stars on one side of the line.",
            "As the line rotates, stars stay on their side.",
            "During a hand-off, the count remains the same.",
            "This invariant property is the key to the proof.",
            "No matter the pivot, the balance never changes."
        ]
        self.setup_layout(title_text, lecture_lines)

        # 2. Colors & Config
        ORANGE_COLOR = "#FFA500"
        LASER_COLOR = YELLOW
        STAR_COLOR = WHITE

        # 3. Mobjects Creation
        # Star positions based on 6x6 grid
        star_a = Star(color=STAR_COLOR, n=5).scale(0.15)
        star_b = Star(color=STAR_COLOR, n=5).scale(0.15)
        star_l1 = Star(color=STAR_COLOR, n=5).scale(0.15)
        star_l2 = Star(color=STAR_COLOR, n=5).scale(0.15)
        star_r1 = Star(color=STAR_COLOR, n=5).scale(0.15)

        # Apply positioning and scaling from Issues 32 & 33
        self.place_at_grid(star_a, "C3", scale_factor=0.8)
        self.place_at_grid(star_b, "B4", scale_factor=0.8)
        self.place_at_grid(star_l1, "B2", scale_factor=0.8)
        self.place_at_grid(star_l2, "C2", scale_factor=0.8)
        self.place_at_grid(star_r1, "C5", scale_factor=0.8)
        
        stars = VGroup(star_a, star_b, star_l1, star_l2, star_r1)
        
        # Labels
        label_a = Text("A", font_size=18).next_to(star_a, DOWN, buff=0.1)
        label_b = Text("B", font_size=18).next_to(star_b, UP, buff=0.1)
        labels = VGroup(label_a, label_b)

        # Counter (Issue 31)
        counter_label = Text("Stars on Left: ", font_size=20)
        counter_val = Integer(2, font_size=20)
        counter_group = VGroup(counter_label, counter_val).arrange(RIGHT, buff=0.1)
        self.place_at_grid(counter_group, 'A6', scale_factor=0.6)

        # Laser Line setup
        line_length = 8
        angle_tracker = ValueTracker(0)
        pivot_pos = [star_a.get_center()]

        laser = Line(
            pivot_pos[0] + UP * line_length/2,
            pivot_pos[0] + DOWN * line_length/2,
            color=LASER_COLOR,
            stroke_width=3
        )

        def update_laser(m):
            angle = angle_tracker.get_value()
            direction = rotate_vector(UP, angle)
            center = pivot_pos[0]
            m.set_points_by_ends(
                center - direction * line_length/2,
                center + direction * line_length/2
            )

        laser.add_updater(update_laser)

        # === Animation for Lecture Line 1 ===
        # Highlight all stars currently to the left of the rotating laser in orange.
        self.lecture[0].set_color(ORANGE_COLOR)
        self.play(FadeIn(stars), FadeIn(labels), run_time=1)
        self.play(Create(laser), run_time=1)
        self.play(
            star_l1.animate.set_color(ORANGE_COLOR),
            star_l2.animate.set_color(ORANGE_COLOR),
            FadeIn(counter_group),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # As the line rotates towards star B, show that star B is the next point the beam will encounter.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(ORANGE_COLOR)
        # Vector AB is [1, 1], so angle from UP is -PI/4.
        self.play(angle_tracker.animate.set_value(-PI/4), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # During a hand-off, the count remains the same.
        # Pivot shift to B.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(ORANGE_COLOR)
        pivot_pos[0] = star_b.get_center()
        self.play(
            Indicate(counter_group, color=ORANGE_COLOR),
            Indicate(star_l1, color=ORANGE_COLOR),
            Indicate(star_l2, color=ORANGE_COLOR),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # After the pivot shift to star B, show that the set of orange stars on the left has not changed.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(ORANGE_COLOR)
        # Rotate slightly more clockwise to show A moving to the right.
        self.play(angle_tracker.animate.set_value(-PI/4 - 0.5), run_time=2)
        self.play(
            Indicate(star_l1, color=ORANGE_COLOR), 
            Indicate(star_l2, color=ORANGE_COLOR),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Display a counter 'Stars on Left: n' at the top, which stays constant during the pivot hand-off.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(ORANGE_COLOR)
        self.play(Indicate(counter_group, color=ORANGE_COLOR), run_time=2)
        # Continue rotation
        self.play(angle_tracker.animate.set_value(-PI/2), run_time=2)
        self.wait(2)
