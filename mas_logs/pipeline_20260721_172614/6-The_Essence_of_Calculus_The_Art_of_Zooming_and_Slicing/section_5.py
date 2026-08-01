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
        # Data from storyboard
        title = "The Limit: Reaching the Infinite"
        lines = [
            "A limit describes what happens as we approach infinity.",
            "It makes our zooms infinitely deep and slices infinitely thin.",
            "This process turns rough approximations into exact mathematical truths."
        ]
        
        self.setup_layout(title, lines)
        
        # Initial dimmed state for lecture lines to make color change visible
        for line in self.lecture:
            line.set_color(GRAY_D)

        # === Animation for Lecture Line 1 ===
        # Show a circle (#FFFFFF) and an inscribed pentagon (#FF4500).
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        circle = Circle(radius=2, color="#FFFFFF")
        # Fix for Issue 36: Shifting to area A2-F6 and scaling to 0.8 to avoid occluding lecture text
        self.place_in_area(circle, "A2", "F6", scale_factor=0.8)
        
        # Create pentagon matching the circle's placement
        polygon = RegularPolygon(n=5, radius=2, color="#FF4500")
        self.place_in_area(polygon, "A2", "F6", scale_factor=0.8)
        
        self.play(Create(circle), Create(polygon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Rapidly increase polygon sides (5 -> 100) toward the circle.
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        
        sides_tracker = ValueTracker(5)
        center = circle.get_center()
        radius = 2 * 0.8 # Matches the scale factor from place_in_area
        
        def polygon_updater(m):
            n = int(sides_tracker.get_value())
            # Manually update points to follow persistent mobject constraint
            points = [
                center + np.array([radius * np.cos(TAU * i / n), radius * np.sin(TAU * i / n), 0])
                for i in range(n)
            ]
            m.set_points_as_corners([*points, points[0]])

        polygon.add_updater(polygon_updater)
        
        # Smooth transition to 100 sides as requested by "rapidly increase"
        self.play(sides_tracker.animate.set_value(100), run_time=4, rate_func=linear)
        polygon.remove_updater(polygon_updater)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Flash the final circle to signify an 'exact' truth.
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        
        # Visual highlight of the circle
        self.play(Flash(circle, color="#FFFFFF", line_length=0.4, flash_radius=radius + 0.2))
        self.wait(2)
