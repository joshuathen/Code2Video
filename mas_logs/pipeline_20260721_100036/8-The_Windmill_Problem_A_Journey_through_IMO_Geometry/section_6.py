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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup layout
        title = "Conclusion: Infinite Connectivity"
        lines = [
            "- The invariant forces the line to visit every point.",
            "- Each point eventually serves as the windmill's pivot.",
            "- Geometry and combinatorics unite to reveal this elegant pattern."
        ]
        self.setup_layout(title, lines)

        # === Initialization of Elements ===
        # Define pivot points (fireflies)
        pivots = VGroup()
        dot_locs = ["C2", "B4", "A5", "D6", "F4", "E2"]
        for loc in dot_locs:
            dot = Dot(color=BLUE, radius=0.08)
            self.place_at_grid(dot, loc)
            pivots.add(dot)

        # Issue 41: Reposition leftmost pivot
        left_pivot = pivots[0]
        self.place_at_grid(left_pivot, 'C2', scale_factor=0.6)

        # Load asset - Issue 26
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/windmill.svg
        windmill_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/windmill.svg")
        windmill_icon.set_color(WHITE).scale(0.15)
        
        # Windmill line
        line = Line(LEFT * 1.5, RIGHT * 1.5, color=WHITE, stroke_width=2)
        
        # Group for right-side positioning - Issue 40
        windmill_system = VGroup(pivots, line, windmill_icon)
        self.place_in_area(windmill_system, 'A2', 'F6', scale_factor=0.7)
        
        # State trackers
        self.current_pivot_idx = 0
        angle_tracker = ValueTracker(0)

        # Add Updaters for the dance
        def line_updater(l):
            angle = angle_tracker.get_value()
            pivot_pos = pivots[self.current_pivot_idx].get_center()
            direction = np.array([np.cos(angle), np.sin(angle), 0])
            l.set_points_by_ends(pivot_pos - 1.8 * direction, pivot_pos + 1.8 * direction)
        
        line.add_updater(line_updater)
        windmill_icon.add_updater(lambda i: i.move_to(pivots[self.current_pivot_idx].get_center()))

        self.add(windmill_system)

        # === Animation for Lecture Line 1 ===
        # Speed up rotation through multiple points to show the 'Dance'.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Rotation sequence through all points
        rotation_angles = [PI/3, 2*PI/3, PI, 4*PI/3, 5*PI/3, 2*PI]
        for i in range(1, 6):
            self.play(
                angle_tracker.animate.set_value(rotation_angles[i-1]),
                run_time=0.4,
                rate_func=linear
            )
            self.current_pivot_idx = i
        
        # Return to start pivot
        self.play(
            angle_tracker.animate.set_value(2*PI),
            run_time=0.4,
            rate_func=linear
        )
        self.current_pivot_idx = 0
        
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Each point eventually serves as the windmill's pivot.
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        # Draw the connection path (#FFFF00) formed by the pivots.
        path_points = [dot.get_center() for dot in pivots]
        path_points.append(path_points[0]) # Close loop
        
        connectivity_poly = VMobject(color="#FFFF00", stroke_width=4)
        connectivity_poly.set_points_as_corners(path_points)
        
        # Show points 'glowing' as they are visited
        glow_dots = pivots.copy().set_color(YELLOW).set_stroke(width=10, opacity=0.4)
        
        self.play(Create(connectivity_poly), FadeIn(glow_dots), run_time=2.0)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Geometry and combinatorics unite to reveal this elegant pattern.
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # Fade out all elements except the final connected path.
        final_path = connectivity_poly.copy()
        
        self.play(
            FadeOut(windmill_system),
            FadeOut(glow_dots),
            FadeOut(connectivity_poly)
        )
        
        # Fix Issue 42: Final connectivity polygon centered in B2-E5
        self.add(final_path)
        self.play(
            final_path.animate.move_to(self.get_area_center("B2", "E5")).scale(1.2).set_stroke(width=6),
            run_time=1.5
        )
        
        # Final emphasis
        self.play(Indicate(final_path, color="#FFFF00", scale_factor=1.1))
        
        self.wait(3)

    def get_area_center(self, top_left, bottom_right):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        return (tl_pos + br_pos) / 2
