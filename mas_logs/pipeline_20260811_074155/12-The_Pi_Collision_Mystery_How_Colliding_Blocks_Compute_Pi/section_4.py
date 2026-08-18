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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Each collision moves our point along the circular path.",
            "Block-wall collisions reflect the point across an axis.",
            "Block-block collisions reflect the point across a tilted line.",
            "These reflections bounce the point around the circle's edge.",
            "Counting collisions is like counting bounces inside a mirror."
        ]
        self.setup_layout("Collisions as Bounces on a Circle", lecture_lines)

        # Colors
        CIRCLE_COLOR = "#ADFF2F"
        POINT_COLOR = "#FF00FF"
        LASER_COLOR = "#FF0000"
        AXIS_COLOR = "#5DADE2" 
        TILTED_LINE_COLOR = ORANGE

        # Area for circle: A3 to F6 center (Issue 36: Improve layout and visibility)
        # We use a larger radius to better utilize the expanded vertical space.
        circle = Circle(radius=2.2, color=CIRCLE_COLOR)
        self.place_in_area(circle, "A3", "F6")
        center = circle.get_center()
        radius = circle.radius

        # Initial geometry
        start_angle = 160 * DEGREES
        p0_pos = center + np.array([radius * np.cos(start_angle), radius * np.sin(start_angle), 0])
        point = Dot(p0_pos, color=POINT_COLOR, radius=0.1)
        
        # === Animation for Lecture Line 1 ===
        # Matching color with circle/point
        self.lecture[0].set_color(CIRCLE_COLOR)
        self.play(Create(circle), FadeIn(point))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(AXIS_COLOR) # Matching color with axis
        
        # Horizontal axis line
        h_axis = Line(center + LEFT*2.5, center + RIGHT*2.5, color=AXIS_COLOR, stroke_width=3)
        self.play(Create(h_axis))
        
        # Calculate reflection across horizontal axis (v -> -v in y-velocity)
        p1_angle = -start_angle
        p1_pos = center + np.array([radius * np.cos(p1_angle), radius * np.sin(p1_angle), 0])
        chord1 = Line(point.get_center(), p1_pos, color=WHITE, stroke_opacity=0.8, stroke_width=2)
        
        self.play(point.animate.move_to(p1_pos), Create(chord1), run_time=1.2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(TILTED_LINE_COLOR) # Matching color with tilted line
        
        # Tilted line (represents the block-block collision reflection line)
        tilted_angle = 20 * DEGREES
        tilted_line = Line(
            center + np.array([2.5*np.cos(tilted_angle), 2.5*np.sin(tilted_angle), 0]),
            center + np.array([-2.5*np.cos(tilted_angle), -2.5*np.sin(tilted_angle), 0]),
            color=TILTED_LINE_COLOR,
            stroke_width=3
        )
        self.play(Create(tilted_line))
        
        # Reflect p1 across tilted line
        p2_angle = 2 * tilted_angle - p1_angle
        p2_pos = center + np.array([radius * np.cos(p2_angle), radius * np.sin(p2_angle), 0])
        chord2 = Line(point.get_center(), p2_pos, color=WHITE, stroke_opacity=0.8, stroke_width=2)
        
        self.play(point.animate.move_to(p2_pos), Create(chord2), run_time=1.2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Multiple bounces to fill the space and show the zig-zag path
        current_angle = p2_angle
        bounces = VGroup(chord1, chord2)
        
        for i in range(8):
            if i % 2 == 0: # Even steps: reflect across axis
                next_angle = -current_angle
            else: # Odd steps: reflect across tilted line
                next_angle = 2 * tilted_angle - current_angle
                
            next_pos = center + np.array([radius * np.cos(next_angle), radius * np.sin(next_angle), 0])
            new_chord = Line(point.get_center(), next_pos, color=WHITE, stroke_opacity=0.8, stroke_width=2)
            self.play(point.animate.move_to(next_pos), Create(new_chord), run_time=0.3)
            bounces.add(new_chord)
            current_angle = next_angle
            
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(LASER_COLOR) # Matching color with laser
        
        # Highlight with laser effect
        laser_bounces = bounces.copy().set_color(LASER_COLOR).set_stroke(width=4, opacity=1)
        self.play(Create(laser_bounces), run_time=2)
        self.wait(2)
