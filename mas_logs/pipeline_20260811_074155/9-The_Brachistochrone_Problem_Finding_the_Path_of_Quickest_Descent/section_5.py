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
        title_text = "The Solution: The Cycloid"
        lecture_lines = [
            "The resulting optimal path is called a cycloid.",
            "A cycloid is traced by a point on a rolling wheel.",
            "Its initial steepness builds kinetic energy very rapidly.",
            "This speed overcomes the extra distance of the curve.",
            "This \"rolling\" geometry minimizes the total travel time."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Constants and parameters
        # Distance from Col 1 to Col 6 is 5 units. 
        # For one rotation (2*PI), we need r = 5 / (2*PI) approx 0.796.
        r = 5.0 / (2 * PI)
        start_center_x = self.grid["A1"][0]
        ceiling_y = self.grid["A1"][1]
        
        theta_tracker = ValueTracker(0)

        # === Animation for Lecture Line 1 ===
        # Show a circle (#FFFFFF) with a red point P (#FF0000) on its rim.
        self.lecture[0].set_color(YELLOW)
        
        ceiling = Line(self.grid["A1"], self.grid["A6"], color=WHITE)
        
        circle = Circle(radius=r, color=WHITE)
        point_p = Dot(color="#FF0000", radius=0.08)
        radial_line = Line(color=WHITE, stroke_width=2)

        # Updaters for the rolling motion
        def update_circle(c):
            theta = theta_tracker.get_value()
            c.move_to([start_center_x + r * theta, ceiling_y - r, 0])
            
        def update_point(p):
            theta = theta_tracker.get_value()
            # Equations for cycloid rolling along y = ceiling_y
            # x = start_x + r*theta - r*sin(theta)
            # y = ceiling_y - r + r*cos(theta)
            # Note: at theta=0, point is at (start_x, ceiling_y)
            p.move_to([
                start_center_x + r * theta - r * np.sin(theta),
                ceiling_y - r + r * np.cos(theta),
                0
            ])

        def update_radial(rl):
            rl.put_start_and_end_on(circle.get_center(), point_p.get_center())

        circle.add_updater(update_circle)
        point_p.add_updater(update_point)
        radial_line.add_updater(update_radial)

        # Initialize positions
        update_circle(circle)
        update_point(point_p)
        update_radial(radial_line)

        self.add(ceiling)
        self.play(Create(circle), Create(point_p), Create(radial_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate the circle rolling along a horizontal line (the ceiling).
        # Trace the path of point P to form a cycloid curve (#FF00FF).
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # TracedPath for the cycloid
        cycloid_path = TracedPath(point_p.get_center, stroke_color="#FF00FF", stroke_width=4)
        self.add(cycloid_path)

        # Roll the wheel for one full rotation
        self.play(theta_tracker.animate.set_value(2 * PI), run_time=6, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight the steep vertical start of the cycloid with a green arrow (#00FF00).
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # The cycloid starts at A1. We place an arrow pointing towards the start region.
        # Start of arrow at B1, end near A1
        start_arrow = self.grid["B1"]
        end_arrow = self.grid["A1"] + DOWN * 0.2
        arrow = Arrow(start=start_arrow, end=end_arrow, color="#00FF00", buff=0.1)
        
        self.play(GrowArrow(arrow))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # This speed overcomes the extra distance of the curve.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Flash the path to emphasize "extra distance"
        self.play(Flash(cycloid_path, color="#FF00FF", line_length=0.3, num_lines=12))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Label the final curve 'The Cycloid: Brachistochrone Solution' (#FFFFFF).
        # Fix: Positioning based on issue 32
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        label = Text("The Cycloid: Brachistochrone Solution", font_size=20, color=WHITE)
        # Using place_in_area as requested in Issue 32
        self.place_in_area(label, 'F3', 'F6', scale_factor=0.7)
        
        self.play(Write(label))
        self.wait(3)
        
        # Cleanup
        self.lecture[4].set_color(WHITE)
        self.wait(1)
