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
        # Initialize the layout with the specific title and lecture lines
        self.setup_layout("Introduction: The Mystery of the Runner", 
                          ["Dash the Cheetah is on the move.", 
                           "His speedometer shows his current speed.", 
                           "His GPS tracks his total distance covered."])
        
        # === Animation for Lecture Line 1 ===
        # Dash the Cheetah [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg] 
        # appears on the track in the center of the screen.
        dash = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        # Fix for Issue 21: Position dash at D1-E3 to avoid overlap.
        self.place_in_area(dash, "D1", "E3", scale_factor=0.8)
        
        self.play(
            FadeIn(dash),
            self.lecture[0].animate.set_color("#FFFF00")
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The screen splits: a Speedometer (#FFD700) appears on the left and a GPS Tracker (#00BFFF) appears on the right.
        
        # Speedometer representation (#FFD700)
        speedometer_arc = Arc(radius=0.6, start_angle=0, angle=PI, color="#FFD700")
        speedometer_needle = Line(speedometer_arc.get_center(), speedometer_arc.get_center() + UP * 0.5, color="#FFD700")
        speedometer_label = Text("Speedometer", font_size=18, color="#FFD700").next_to(speedometer_arc, DOWN, buff=0.1)
        speedometer = VGroup(speedometer_arc, speedometer_needle, speedometer_label)
        
        # GPS Tracker representation (#00BFFF)
        gps_rect = RoundedRectangle(height=1.2, width=0.8, corner_radius=0.1, color="#00BFFF")
        gps_dot = Dot(color=RED, radius=0.05).move_to(gps_rect.get_center())
        gps_path = Line(gps_rect.get_bottom() + UP*0.2, gps_rect.get_top() - UP*0.2, color="#00BFFF").set_stroke(opacity=0.5)
        gps_label = Text("GPS Tracker", font_size=18, color="#00BFFF").next_to(gps_rect, DOWN, buff=0.1)
        gps_tracker = VGroup(gps_rect, gps_dot, gps_path, gps_label)
        
        # Fix for Issue 23: Reposition and scale speedometer and gps_tracker.
        self.place_in_area(speedometer, "A1", "B2", scale_factor=0.8)
        self.place_in_area(gps_tracker, "A5", "B6", scale_factor=0.8)
        
        # Fix for Issue 22: Position dash_target at D4-E6 to avoid clutter.
        target_center = self.grid["D5"] # Approximating D4-E6 center
        
        self.play(
            dash.animate.move_to(target_center).scale(0.6),
            FadeIn(speedometer),
            FadeIn(gps_tracker),
            self.lecture[1].animate.set_color("#FFFF00")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Pulsing arrows and question marks appear between the Speedometer and GPS, linking the two.
        
        # Arrows linking the two devices
        arrow_r = Arrow(speedometer.get_right(), gps_tracker.get_left(), buff=0.2, color=WHITE, stroke_width=2)
        arrow_l = Arrow(gps_tracker.get_left(), speedometer.get_right(), buff=0.2, color=WHITE, stroke_width=2).shift(DOWN*0.3)
        arrow_r.shift(UP*0.3)
        
        q_mark = Text("?", font_size=36, color=WHITE)
        # Position question mark between the two (Issue 23 mentioned central question mark)
        self.place_in_area(q_mark, "B3", "C4")
        
        arrows_group = VGroup(arrow_r, arrow_l, q_mark)
        
        self.play(
            Create(arrow_r),
            Create(arrow_l),
            Write(q_mark),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        
        # Pulsing effect for the link
        self.play(
            arrows_group.animate.scale(1.15),
            rate_func=there_and_back,
            run_time=2
        )
        self.wait(2)
