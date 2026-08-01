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
        title_text = "The 180-Degree Flip"
        lecture_lines = [
            "Now, let the line rotate exactly 180 degrees.",
            "The line returns to its original position in space.",
            "However, its \"left\" and \"right\" sides have now swapped.",
            "Points originally on the left are now on the right.",
            "This requires the sets on both sides to be equal."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        CYAN = "#00FFFF"
        MAGENTA = "#FF00FF"
        WHITE_COLOR = "#FFFFFF"

        # Grid-based positions
        pivot_pos = (self.grid["C3"] + self.grid["D4"]) / 2
        
        # Initial Points
        # Points on top and bottom to show swapping sides
        points_top = VGroup(*[
            Dot(pivot_pos + np.array([x, 0.8, 0]), radius=0.1) for x in [-1.5, 0, 1.5]
        ])
        points_bottom = VGroup(*[
            Dot(pivot_pos + np.array([x, -0.8, 0]), radius=0.1) for x in [-1.5, 0, 1.5]
        ])
        
        # Line
        line_length = 5.0
        line = Line(
            pivot_pos + LEFT * (line_length/2),
            pivot_pos + RIGHT * (line_length/2),
            color=CYAN,
            stroke_width=6
        )
        
        # Pivot point dot
        pivot_dot = Dot(pivot_pos, color=WHITE_COLOR, radius=0.1)

        # Angle Tracker
        angle_tracker = ValueTracker(0)

        # Updaters
        def update_line(l):
            angle = angle_tracker.get_value() * DEGREES
            l.set_points_by_ends(
                pivot_pos + rotate_vector(LEFT * (line_length/2), angle),
                pivot_pos + rotate_vector(RIGHT * (line_length/2), angle)
            )
        
        def update_point_color(m, p_pos):
            angle = angle_tracker.get_value() * DEGREES
            # The normal vector defines the "side" relative to the line's direction
            normal = rotate_vector(UP, angle)
            vec = p_pos - pivot_pos
            # Dot product determines which side of the line the point is on
            if np.dot(vec, normal) > 0:
                m.set_color(CYAN)
            else:
                m.set_color(MAGENTA)

        line.add_updater(update_line)
        for p in points_top:
            p.add_updater(lambda m, p=p: update_point_color(m, p.get_center()))
        for p in points_bottom:
            p.add_updater(lambda m, p=p: update_point_color(m, p.get_center()))

        # === Animation for Lecture Line 1 ===
        # Now, let the line rotate exactly 180 degrees.
        self.lecture[0].set_color(CYAN)
        self.add(line, pivot_dot, points_top, points_bottom)
        
        initial_label = Text("Initial Orientation", font_size=20, color=CYAN)
        self.place_at_grid(initial_label, "A3", scale_factor=0.8) # Fixed Issue 38: Moved to A3
        self.play(FadeIn(initial_label))
        
        # Continuous rotation through 180 degrees
        self.play(angle_tracker.animate.set_value(180), run_time=6, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # The line returns to its original position in space.
        self.lecture[1].set_color(CYAN)
        
        final_label = Text("180° Rotation (Same Line)", font_size=20, color=MAGENTA)
        self.place_at_grid(final_label, "F3", scale_factor=0.8) # Fixed Issue 39: Moved to F3
        
        self.play(FadeIn(final_label))
        self.play(Indicate(line, color=CYAN))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # However, its "left" and "right" sides have now swapped.
        self.lecture[2].set_color(MAGENTA)
        
        # Show swap with visuals
        arrow_swap = DoubleArrow(pivot_pos + UP*1.2, pivot_pos + DOWN*1.2, color=WHITE_COLOR, buff=0.1)
        
        # Change line color to MAGENTA to highlight the "swapped" state
        self.play(line.animate.set_color(MAGENTA))
        self.play(Create(arrow_swap))
        self.wait(0.5)
        self.play(FadeOut(arrow_swap))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Points originally on the left are now on the right.
        self.lecture[3].set_color(CYAN)
        
        # Points have already updated colors via updaters. Emphasize them.
        self.play(
            *[Flash(p, color=p.get_color(), flash_radius=0.4) for p in points_top],
            *[Flash(p, color=p.get_color(), flash_radius=0.4) for p in points_bottom],
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This requires the sets on both sides to be equal.
        self.lecture[4].set_color(WHITE)
        
        # Highlight groups to show they are "equal" in concept
        box_cyan_side = SurroundingRectangle(points_bottom, color=CYAN, buff=0.2)
        box_magenta_side = SurroundingRectangle(points_top, color=MAGENTA, buff=0.2)
        
        self.play(Create(box_cyan_side), Create(box_magenta_side))
        
        # Final pulse effect
        self.play(
            line.animate.scale(1.1).set_stroke(width=10),
            box_cyan_side.animate.scale(1.05),
            box_magenta_side.animate.scale(1.05),
            rate_func=there_and_back,
            run_time=2
        )
        self.wait(2)

        # Cleanup updaters
        line.clear_updaters()
        for p in points_top:
            p.clear_updaters()
        for p in points_bottom:
            p.clear_updaters()
