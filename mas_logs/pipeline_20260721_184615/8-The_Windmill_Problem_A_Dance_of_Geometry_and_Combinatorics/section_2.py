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
        # Initialization
        title = "Prerequisite: The Rules of the Game"
        lines = [
            "No three points lie on a single straight line.",
            "The line rotates until it strikes exactly one point.",
            "The pivot immediately shifts to this new point."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        WHITE_COLOR = "#FFFFFF"
        GOLD_COLOR = "#FFD700"
        GREEN_COLOR = "#00FF00"
        RED_COLOR = "#FF0000"
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(RED_COLOR)
        
        # Show 3 points in a triangle - Fix per Issue 25
        p1 = Dot(color=WHITE_COLOR)
        p2 = Dot(color=WHITE_COLOR)
        p3 = Dot(color=WHITE_COLOR)
        
        # Triangle positions from Issue 25
        self.place_at_grid(p1, "B4")
        self.place_at_grid(p2, "C6")
        self.place_at_grid(p3, "D4")
        
        points = VGroup(p1, p2, p3)
        self.play(FadeIn(points))
        self.wait(1.0)
        
        # Move to collinear position (B4, B5, B6) to demonstrate the "invalid" condition
        # This keeps the objects further from the lecture area per Issue 25
        self.play(
            p1.animate.move_to(self.grid["B4"]),
            p2.animate.move_to(self.grid["B5"]),
            p3.animate.move_to(self.grid["B6"]),
            run_time=1.5
        )
        
        # Draw invalid line and X
        # Line from B3 to B6+ (x=2.5 to x=6.0)
        invalid_line = Line(self.grid["B3"], self.grid["B6"] + RIGHT * 0.5, color=RED_COLOR)
        cross = VGroup(
            Line(LEFT + UP, RIGHT + DOWN, color=RED_COLOR),
            Line(LEFT + DOWN, RIGHT + UP, color=RED_COLOR)
        ).scale(0.3).move_to(self.grid["B5"])
        
        self.play(Create(invalid_line))
        self.play(Create(cross))
        self.wait(2.0)
        
        # Clean up
        self.play(FadeOut(points), FadeOut(invalid_line), FadeOut(cross))
        self.lecture[0].set_color(WHITE_COLOR)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GOLD_COLOR)
        self.wait(1.5) 
        
        # Setup for rotation - Fix per Issue 23 and 24
        pivot_dot = Dot(color=GOLD_COLOR)
        target_dot = Dot(color=WHITE_COLOR)
        other_dot = Dot(color=WHITE_COLOR)
        
        # Positioned far enough right to avoid overlap
        self.place_at_grid(pivot_dot, "C4")  # Issue 23 fix
        self.place_at_grid(target_dot, "D6") # Issue 24 fix
        self.place_at_grid(other_dot, "B6")  # Issue 24 fix
        
        # Rotation mechanics
        # Starting angle slightly counter-clockwise from target (approx 0.5 rad above horiz)
        angle_tracker = ValueTracker(0.5) 
        
        # The line rotates clockwise around a single pivot point
        # Use length 6 (3 each way) so centered at C4 (x=3.5) it spans x=0.5 to x=6.5
        windmill_line = Line(LEFT * 3, RIGHT * 3, color=GOLD_COLOR)
        windmill_line.add_updater(lambda m: m.set_angle(angle_tracker.get_value()).move_to(pivot_dot.get_center()))
        
        self.play(FadeIn(pivot_dot), FadeIn(target_dot), FadeIn(other_dot))
        self.add(windmill_line)
        self.wait(1.0)
        
        # Target angle calculation (D6 relative to C4)
        # C4 is (3.5, 0.2), D6 is (5.5, -0.8)
        # dx = 2.0, dy = -1.0
        target_angle = np.arctan2(-1.0, 2.0)
        
        # Rotate clockwise (decrease angle) using rate_functions.linear per [L024]
        self.play(angle_tracker.animate.set_value(target_angle), run_time=3, rate_func=rate_functions.linear)
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN_COLOR)
        self.wait(2.0)
        
        # Highlight intersection using Indicate per [L004]
        self.play(Indicate(target_dot, color=GREEN_COLOR))
        self.wait(0.5)
        
        # Pivot immediately shifts to this new point
        self.play(
            pivot_dot.animate.move_to(target_dot.get_center()),
            windmill_line.animate.set_color(GREEN_COLOR),
            FadeOut(target_dot, run_time=0.5),
            run_time=1.5
        )
        
        # New line rotation around the new pivot
        windmill_line.clear_updaters()
        windmill_line.add_updater(lambda m: m.set_angle(angle_tracker.get_value()).move_to(pivot_dot.get_center()))
        
        # Continue clockwise rotation
        self.play(angle_tracker.animate.set_value(target_angle - 1.0), run_time=2, rate_func=rate_functions.linear)
        self.wait(2.0)
        
        # Final cleanup/reset colors
        self.lecture[1].set_color(WHITE_COLOR)
        self.lecture[2].set_color(WHITE_COLOR)
