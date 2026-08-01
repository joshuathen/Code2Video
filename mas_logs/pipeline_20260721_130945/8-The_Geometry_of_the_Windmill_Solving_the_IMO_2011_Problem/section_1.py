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
        title = "The Hook: Meet Milly the Laser Robot"
        lecture_lines = [
            "Meet Milly, a robot in a field of stars.",
            "She holds a laser beam that rotates clockwise.",
            "Today, we explore how this beam moves."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # Colors
        STAR_COLOR = WHITE
        LASER_COLOR = TEAL
        HIGHLIGHT_COLOR = YELLOW
        
        # === Animation for Lecture Line 1 ===
        # L003: Optimize layout. Use Cols 3-5 to avoid lecture area (Issues 22, 23).
        # Star field positions. Pivot 1 at D5, Pivot 2 at C4.
        star_positions = ["D5", "C4", "B4", "C6", "B5", "F5", "D3"]
        stars = VGroup(*[Dot(color=STAR_COLOR, radius=0.1) for _ in star_positions])
        for star, pos in zip(stars, star_positions):
            self.place_at_grid(star, pos)
            
        # Label for Milly
        # L003: Labeled object scaled to 0.7. Positioned at D4 (1 unit from D5).
        milly_label = Text("Milly", font_size=24, color=HIGHLIGHT_COLOR)
        self.place_at_grid(milly_label, "D4", scale_factor=0.7)

        self.play(
            self.lecture[0].animate.set_color(WHITE),
            FadeIn(stars, lag_ratio=0.1),
            Write(milly_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # She holds a laser beam that rotates clockwise.
        # Track the pivot and rotation angle
        angle_tracker = ValueTracker(180 * DEGREES) # Starting horizontal (pointing Left)
        
        # L008: Use a persistent mobject (Dot) to track the pivot center for the updater.
        pivot_tracker_dot = Dot(self.grid["D5"]).set_opacity(0)
        
        def get_laser_endpoints():
            p = pivot_tracker_dot.get_center()
            theta = angle_tracker.get_value()
            direction = np.array([np.cos(theta), np.sin(theta), 0])
            # Extend 7 units in both directions to span the scene
            return p - direction * 7, p + direction * 7

        laser = Line(color=LASER_COLOR, stroke_width=3)
        # Initial position setup
        l_start, l_end = get_laser_endpoints()
        laser.put_start_and_end_on(l_start, l_end)
        
        # L008/L011: Use updater for stable animated movement. 
        laser.add_updater(lambda m: m.put_start_and_end_on(*get_laser_endpoints()))
        
        self.play(
            self.lecture[1].animate.set_color(LASER_COLOR),
            Create(laser),
            run_time=1
        )
        
        # Clockwise rotation: 180 deg (West) to 135 deg (North-West)
        # In Manim, decreasing the angle value results in clockwise movement from this start point.
        self.play(
            angle_tracker.animate.set_value(135 * DEGREES),
            run_time=2,
            rate_func=linear
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Today, we explore how this beam moves.
        self.play(self.lecture[2].animate.set_color(HIGHLIGHT_COLOR))
        
        # Highlight the hit star at C4 (L004: Indicate)
        self.play(Indicate(stars[1], color=HIGHLIGHT_COLOR, scale_factor=1.5))
        
        # Shift pivot to star 2 (C4) and move Milly label to C3
        # Rotate further clockwise from 135 towards 45.
        self.play(
            pivot_tracker_dot.animate.move_to(self.grid["C4"]),
            milly_label.animate.move_to(self.grid["C3"]),
            angle_tracker.animate.set_value(45 * DEGREES),
            run_time=3,
            rate_func=linear
        )
        self.wait(2)
