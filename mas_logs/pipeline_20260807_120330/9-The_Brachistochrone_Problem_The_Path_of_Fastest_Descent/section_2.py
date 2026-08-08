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
        # 1. Setup layout
        title_text = "Prerequisite Knowledge: The Speed-Height Relationship"
        lecture_lines = [
            "Velocity depends on the vertical distance dropped.",
            "Energy conservation dictates speed increases with depth.",
            "A steeper initial drop grants higher velocity early."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Define Colors
        COLOR_FORMULA = "#FFFF00"  # bright yellow
        COLOR_VECTOR = "#00FFFF"   # cyan
        COLOR_SPEEDO = "#FFA500"   # orange
        
        # Constants for physics
        g = 9.8
        
        # === Animation for Lecture Line 1 ===
        # "Velocity depends on the vertical distance dropped."
        self.play(self.lecture[0].animate.set_color(COLOR_FORMULA))
        
        formula = MathTex("v = \\sqrt{2gy}", color=COLOR_FORMULA)
        # Resolved Issue 25: Adjust formula area and scale
        self.place_in_area(formula, 'B2', 'B4', scale_factor=1.1)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Energy conservation dictates speed increases with depth."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_VECTOR)
        )
        
        # Visual setup for falling block
        # Start and end points for the fall
        start_pos = self.grid["C3"].copy()
        end_pos = self.grid["E3"].copy()
        dist = start_pos[1] - end_pos[1]
        
        block = Square(side_length=0.4, fill_opacity=1, color=WHITE)
        self.place_at_grid(block, "C3")
        
        y_tracker = ValueTracker(0)
        block.add_updater(lambda m: m.move_to(start_pos + DOWN * y_tracker.get_value()))
        
        # Velocity vector that grows with the block (v = sqrt(2gy))
        # Optimized: Using persistent Vector with add_updater instead of always_redraw
        moving_vector = Vector(DOWN * 0.1, color=COLOR_VECTOR)
        moving_vector.move_to(block.get_center(), aligned_edge=UP)
        
        def update_vector(v):
            y_val = y_tracker.get_value()
            v_mag = np.sqrt(2 * g * y_val) * 0.2 + 0.1
            curr_center = block.get_center()
            v.put_start_and_end_on(curr_center, curr_center + DOWN * v_mag)

        moving_vector.add_updater(update_vector)
        
        trail_vectors = VGroup()
        self.add(block, moving_vector, trail_vectors)
        
        # Animation: Fall to demonstrate speed increasing with depth
        self.play(
            y_tracker.animate.set_value(dist),
            run_time=3,
            rate_func=rate_functions.ease_in_quad # Simple acceleration simulation
        )
        
        # Add trail static vectors at fixed height intervals
        for y_v in [0.4, 0.8, 1.2, 1.6]:
            if y_v < dist:
                v_mag = np.sqrt(2 * g * y_v) * 0.2 + 0.1
                v_static = Vector(DOWN * v_mag, color=COLOR_VECTOR, stroke_opacity=0.4)
                v_static.move_to(start_pos + DOWN * y_v, aligned_edge=UP)
                trail_vectors.add(v_static)
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "A steeper initial drop grants higher velocity early."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_SPEEDO)
        )
        
        # Speedometer setup
        # Resolved Issue 24: Move speed gauge to D5 for better proximity
        gauge = Arc(radius=0.7, start_angle=PI, angle=-PI, color=WHITE)
        self.place_at_grid(gauge, "D5", scale_factor=0.8)
        gauge_line = Line(gauge.get_start(), gauge.get_end(), color=WHITE)
        
        # Needle rotation based on velocity ratio
        center_gauge = gauge.get_center()
        needle = Line(center_gauge, center_gauge + LEFT * 0.6, color=COLOR_SPEEDO)
        needle.set_stroke(width=4)
        
        speed_label = Text("Speed", font_size=16, color=COLOR_SPEEDO)
        speed_label.next_to(gauge, DOWN, buff=0.1)
        
        speedo = VGroup(gauge, gauge_line, needle, speed_label)
        
        def update_needle(m):
            y_val = y_tracker.get_value()
            v_val = np.sqrt(2 * g * y_val)
            v_max = np.sqrt(2 * g * dist)
            ratio = v_val / v_max if v_max > 0 else 0
            angle = PI * (1 - ratio) # Needle moves from PI (left) to 0 (right)
            
            c = gauge.get_center()
            end = c + np.array([np.cos(angle), np.sin(angle), 0]) * 0.6
            m.put_start_and_end_on(c, end)
            
        needle.add_updater(update_needle)
        
        self.play(FadeIn(speedo))
        
        # Reset and run again to show synchronization between block fall and speedometer
        self.play(
            y_tracker.animate.set_value(0),
            FadeOut(trail_vectors),
            run_time=1
        )
        self.wait(0.5)
        self.play(
            y_tracker.animate.set_value(dist),
            run_time=3,
            rate_func=rate_functions.ease_in_quad
        )
        self.wait(2)
