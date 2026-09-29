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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Complex waveforms emerge from simple rotating circles.",
            "Each circle acts as a basic rotating phasor.",
            "Combining different frequencies creates complex paths.",
            "This is the core of epicycles.",
            "Motion blends into a single smooth curve."
        ]
        self.setup_layout("The Intuition: Epicycles and Circular Motion", lecture_lines)
        
        # Setup Animation Elements
        circle = Circle(radius=2.0, color="#FFFFFF")
        self.place_in_area(circle, 'B3', 'D5', scale_factor=0.85)
        
        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg").scale(0.3)
        compass.move_to(circle.get_center())
        
        watch = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/watch.svg").scale(0.2)
        
        point_p = Dot(circle.point_from_proportion(0), color="#FF00FF")
        watch.add_updater(lambda m: m.next_to(point_p, UP, buff=0.1))
        
        # Track rotation
        tracker = ValueTracker(0)
        
        def update_point(m):
            angle = tracker.get_value()
            m.move_to(circle.point_from_proportion(angle % 1))
            
        point_p.add_updater(update_point)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"), Create(circle), FadeIn(compass), run_time=1)
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"), Create(point_p), FadeIn(watch), run_time=1)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"), tracker.animate.set_value(1), run_time=3, rate_func=linear)
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF8000"), run_time=1)
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFFFF"), run_time=1)
        
        point_p.remove_updater(update_point)
        self.wait(2)
