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
        # Setup title and lecture lines
        lecture_lines = [
            "The laser always rotates clockwise around a pivot.",
            "Striking a new star instantly transfers the center.",
            "Rotation continues seamlessly around the new star."
        ]
        self.setup_layout("The Rule of the Windmill", lecture_lines)

        # --- Assets and Objects ---
        # Star A (initial pivot) - Fixed position to 'C5' to avoid occlusion
        star_a = Dot(color="#FFFF00", radius=0.1)
        self.place_at_grid(star_a, "C5")
        label_a = Text("A", font_size=16, color=WHITE).next_to(star_a, UP + RIGHT, buff=0.1)
        
        # Star B (next pivot) - Fixed position to 'E6' to avoid overlap
        star_b = Dot(color="#00FFFF", radius=0.1)
        self.place_at_grid(star_b, "E6")
        label_b = Text("B", font_size=16, color=WHITE).next_to(star_b, DOWN + RIGHT, buff=0.1)

        # Windmill Asset - Fixed start position to 'C5'
        windmill_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/windmill.svg")
        windmill_icon.set_color(WHITE)
        self.place_at_grid(windmill_icon, "C5", scale_factor=0.3)

        # Rotating Laser Line
        angle_tracker = ValueTracker(0)
        pivot_tracker = Dot(point=self.grid["C5"])
        
        laser_line = Line(LEFT * 6, RIGHT * 6, color="#FF0000", stroke_width=4)
        laser_line.add_updater(lambda m: m.set_angle(angle_tracker.get_value()).move_to(pivot_tracker.get_center()))

        self.add(star_a, star_b, label_a, label_b, laser_line, windmill_icon)

        # Calculate angle to strike Star B from Star A
        # Vector A(C5) to B(E6)
        # C5: [4.5, 0.2], E6: [5.5, -1.8]
        # dx = 1.0, dy = -2.0
        strike_angle = np.arctan2(-2.0, 1.0)

        # === Animation for Lecture Line 1 ===
        # The laser always rotates clockwise around a pivot.
        self.play(self.lecture[0].animate.set_color("#FF0000"), run_time=0.5)
        
        # Initial rotation: Clockwise from horizontal towards B
        self.play(
            angle_tracker.animate.set_value(strike_angle),
            run_time=2,
            rate_func=linear
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Striking a new star instantly transfers the center.
        self.play(self.lecture[1].animate.set_color("#00FFFF"), run_time=0.5)
        
        # Instant transfer of pivot
        self.play(
            star_a.animate.set_color(WHITE),
            star_b.animate.set_color("#FFFF00"),
            windmill_icon.animate.move_to(self.grid["E6"]),
            pivot_tracker.animate.move_to(self.grid["E6"]),
            run_time=0.2
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Rotation continues seamlessly around the new star.
        self.play(self.lecture[2].animate.set_color("#FFFF00"), run_time=0.5)
        
        # Continue clockwise rotation (rotate 120 degrees further)
        self.play(
            angle_tracker.animate.set_value(strike_angle - 120 * DEGREES),
            run_time=2,
            rate_func=linear
        )
        self.wait(2)
