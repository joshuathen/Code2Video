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

class Section2Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title_text = "Prerequisite: Antipodal Points and Continuous Mapping"
        lecture_lines = [
            "Antipodal points are directly opposite on a circle.",
            "Continuous functions map points to values like temperature.",
            "Temperature changes smoothly along the circular wire."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line in blue
        self.play(self.lecture[0].animate.set_color("#00BFFF"))

        # Create Circle - occupy center-right area (Rows A-C, Columns 3-5)
        # Fix for Issue 25: Relocate circle to reduce crowding
        circle = Circle(radius=1.0, color=WHITE)
        self.place_in_area(circle, "A3", "C5")
        
        # Determine exact coordinates for antipodal points on the placed circle
        p1_coord = circle.point_at_angle(0)
        p2_coord = circle.point_at_angle(PI)
        
        dot1 = Dot(p1_coord, color="#00BFFF")
        dot2 = Dot(p2_coord, color="#00BFFF")
        
        # Labels - placed using grid system to avoid manual coordinates
        # Fix for Issue 25: Move labels to B6 and B2
        label1 = Text("Antipodal", font_size=18, color="#00BFFF")
        label2 = Text("Antipodal", font_size=18, color="#00BFFF")
        self.place_at_grid(label1, "B6")
        self.place_at_grid(label2, "B2")

        self.play(Create(circle))
        self.play(FadeIn(dot1, dot2), Write(label1), Write(label2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )

        # Temperature Scale - horizontal line at bottom (Row F)
        # Fix for Issue 27: Reduce scale_factor to 1.0
        temp_line = Line(LEFT, RIGHT, color=WHITE)
        self.place_in_area(temp_line, "F2", "F5", scale_factor=1.0)
        
        low_label = Text("Cold", font_size=16, color=BLUE_A)
        high_label = Text("Hot", font_size=16, color=RED_A)
        self.place_at_grid(low_label, "F1")
        self.place_at_grid(high_label, "F6")

        # Mapping Arrow from circle area to scale area (Row D to Row E)
        mapping_arrow = Arrow(self.grid["D4"], self.grid["E4"], color=WHITE, buff=0.1)
        mapping_text = Text("Continuous Function", font_size=16, color="#FFFFFF")
        # Fix for Issue 26: Center mapping_text in area E3-E5
        self.place_in_area(mapping_text, "E3", "E5")

        self.play(Create(temp_line), Write(low_label), Write(high_label))
        self.play(GrowArrow(mapping_arrow), FadeIn(mapping_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(ORANGE)
        )

        # Tracker for motion around the circle
        theta_tracker = ValueTracker(0)
        
        # Moving dot on the circle representing the thermometer
        moving_dot = Dot(color=ORANGE).scale(1.2)
        moving_dot.add_updater(lambda m: m.move_to(circle.point_at_angle(theta_tracker.get_value())))
        
        # Moving indicator on the temperature scale representing the reading
        indicator = Dot(color=ORANGE).scale(1.2)
        
        def indicator_updater(m):
            # Map circle angle to temperature value using a smooth sine wave
            val = np.sin(theta_tracker.get_value())
            norm_val = (val + 1) / 2
            
            # Interpolate position between F2 and F5 grid centers
            x_start = self.grid["F2"][0]
            x_end = self.grid["F5"][0]
            x_pos = x_start + norm_val * (x_end - x_start)
            y_pos = self.grid["F2"][1]
            m.move_to([x_pos, y_pos, 0])
            
        indicator.add_updater(indicator_updater)

        # Perform the circular motion to demonstrate smooth temperature fluctuation
        self.add(moving_dot, indicator)
        self.play(theta_tracker.animate.set_value(2 * PI), run_time=6, rate_func=linear)
        self.wait(2)
