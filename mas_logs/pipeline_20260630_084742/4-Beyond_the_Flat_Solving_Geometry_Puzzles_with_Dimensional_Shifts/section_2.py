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
        # Setup the scene with title and lecture lines
        lecture_lines = [
            'Cross-sections show slices of higher-dimensional objects.',
            'A sphere passing through Flatland appears as a growing circle.',
            'These slices represent parts of a larger reality.'
        ]
        self.setup_layout("Cross-Sections: Slicing the Higher Reality", lecture_lines)
        
        # Define Colors
        CYAN_COLOR = "#00FFFF"
        WHITE_COLOR = "#FFFFFF"
        
        # === Animation for Lecture Line 1 ===
        # Highlight first line in cyan
        self.play(self.lecture[0].animate.set_color(CYAN_COLOR))
        
        # Create horizontal line (Flatland boundary)
        # Resolved Issue 36: Shorten line (C2 to C6) to avoid notes obstruction
        flatland_line = Line(
            start=self.grid["C2"],
            end=self.grid["C6"],
            color=WHITE_COLOR,
            stroke_width=2
        )
        
        # Create the cyan circle (representing a sphere)
        # Resolved Issue 37: Increase scale to 1.2
        circle_radius = 1.2
        cyan_circle = Circle(radius=1.0, color=CYAN_COLOR, stroke_width=4)
        self.place_in_area(cyan_circle, "A3", "B4", scale_factor=1.2)
        
        self.play(Create(flatland_line), Create(cyan_circle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line in white
        self.play(
            self.lecture[0].animate.set_color(WHITE_COLOR),
            self.lecture[1].animate.set_color(WHITE_COLOR)
        )
        
        # ValueTracker for circle's vertical movement
        start_y = cyan_circle.get_center()[1]
        y_tracker = ValueTracker(start_y)
        
        # Intersection segment on the Flatland line
        intersection_segment = Line(color=WHITE_COLOR, stroke_width=8)
        # Initialize hidden
        intersection_segment.set_alpha(0)
        
        # Positional updaters
        def circle_update(m):
            m.move_to([cyan_circle.get_center()[0], y_tracker.get_value(), 0])
            
        def intersection_update(seg):
            y_c = y_tracker.get_value()
            y_line = flatland_line.get_center()[1]
            dist = abs(y_c - y_line)
            
            if dist < circle_radius:
                # Calculate chord length: 2 * sqrt(R^2 - d^2)
                half_len = np.sqrt(max(0, circle_radius**2 - dist**2))
                center_x = cyan_circle.get_center()[0]
                seg.set_points_as_corners([
                    [center_x - half_len, y_line, 0],
                    [center_x + half_len, y_line, 0]
                ])
                seg.set_alpha(1)
            else:
                seg.set_alpha(0)

        cyan_circle.add_updater(circle_update)
        intersection_segment.add_updater(intersection_update)
        self.add(intersection_segment)
        
        # Move circle slowly downward through the line to Row F
        target_y = self.grid["F3"][1]
        self.play(
            y_tracker.animate.set_value(target_y),
            run_time=8,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third line in Cyan
        self.play(
            self.lecture[1].animate.set_color(WHITE_COLOR),
            self.lecture[2].animate.set_color(CYAN_COLOR)
        )
        
        # Final pause
        self.wait(2)
        
        # Cleanup
        cyan_circle.clear_updaters()
        intersection_segment.clear_updaters()
