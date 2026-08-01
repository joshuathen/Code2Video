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
        # Setup layout with title and lecture lines
        self.setup_layout("Visualizing the Pivot Swap", [
            "Watch as the line approaches the next point nearby.",
            "Upon contact, the rotation axis shifts to that point.",
            "The line's direction remains continuous during this transition."
        ])
        
        # Colors
        COLOR_LINE = "#00FFFF"
        COLOR_A = "#00FF00"
        COLOR_B = "#FF0000"
        COLOR_TRAIL = "#FFFF00"
        COLOR_FLASH = "#FFFFFF"

        # === Initialization of Mobjects ===
        
        # Points A and B
        point_a = Dot(color=COLOR_A)
        point_b = Dot(color=COLOR_B)
        
        # Positioning according to VideoCritic Issues 33 & 34
        # Move point_a to C4 and point_b to E5 for better layout
        self.place_at_grid(point_a, "C4", scale_factor=0.8)
        self.place_at_grid(point_b, "E5", scale_factor=0.8)
        
        # Labels (positioned within 1 grid unit)
        label_a = Text("A", font_size=20, color=COLOR_A)
        label_b = Text("B", font_size=20, color=COLOR_B)
        label_a.next_to(point_a, UP, buff=0.1)
        label_b.next_to(point_b, DOWN, buff=0.1)
        
        # State trackers for rotation and pivot
        # Start at 90 degrees (vertical)
        angle_tracker = ValueTracker(PI/2)
        pivot_tracker = VectorizedPoint(point_a.get_center())
        line_length = 15 # Long line to simulate the infinite windmill line
        
        def get_line_endpoints():
            pivot = pivot_tracker.get_location()
            angle = angle_tracker.get_value()
            direction = np.array([np.cos(angle), np.sin(angle), 0])
            start = pivot - (line_length / 2) * direction
            end = pivot + (line_length / 2) * direction
            return [start, end]

        # Persistent Line object with updater
        line = Line(*get_line_endpoints(), color=COLOR_LINE, stroke_width=4)
        line.add_updater(lambda m: m.put_start_and_end_on(*get_line_endpoints()))
        
        # Invisible dots for tracing the sweep near the pivot
        def get_tip1_pos():
            pivot = pivot_tracker.get_location()
            angle = angle_tracker.get_value()
            direction = np.array([np.cos(angle), np.sin(angle), 0])
            return pivot + 2.5 * direction

        def get_tip2_pos():
            pivot = pivot_tracker.get_location()
            angle = angle_tracker.get_value()
            direction = np.array([np.cos(angle), np.sin(angle), 0])
            return pivot - 2.5 * direction

        tip1 = Dot(radius=0).add_updater(lambda m: m.move_to(get_tip1_pos()))
        tip2 = Dot(radius=0).add_updater(lambda m: m.move_to(get_tip2_pos()))
        
        # Sweep trails (faint Yellow)
        trail1 = TracedPath(tip1.get_center, stroke_color=COLOR_TRAIL, stroke_width=2, stroke_opacity=0.3)
        trail2 = TracedPath(tip2.get_center, stroke_color=COLOR_TRAIL, stroke_width=2, stroke_opacity=0.3)
        
        # === Animation for Lecture Line 1 ===
        # "Watch as the line approaches the next point nearby."
        self.lecture[0].set_color(COLOR_LINE)
        self.add(point_a, point_b, label_a, label_b, line, tip1, tip2, trail1, trail2)
        
        # Calculate angle where the line hits point B
        pos_a = point_a.get_center()
        pos_b = point_b.get_center()
        diff = pos_b - pos_a
        hit_angle = np.arctan2(diff[1], diff[0])
        
        # Rotate clockwise to hit point B
        # PI/2 -> hit_angle is a negative change (clockwise)
        self.play(angle_tracker.animate.set_value(hit_angle), run_time=3, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # "Upon contact, the rotation axis shifts to that point."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_FLASH)
        
        # Flash effect at points A and B to signal pivot swap
        self.play(
            Flash(point_a, color=COLOR_FLASH),
            Flash(point_b, color=COLOR_FLASH),
            run_time=0.8
        )
        
        # Instant pivot swap to point B
        pivot_tracker.move_to(pos_b)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # "The line's direction remains continuous during this transition."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_TRAIL)
        
        # Continue clockwise rotation from point B
        self.play(angle_tracker.animate.set_value(hit_angle - PI/2), run_time=3, rate_func=linear)
        self.wait(1)
        
        # Reset colors of lecture lines
        self.lecture[2].set_color(WHITE)
