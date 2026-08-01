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
        # Fetching data from shared state
        title = "The 180-Degree Flip"
        lecture_lines = [
            "After a half-rotation, the line returns to its angle.",
            "However, the line is now oriented in reverse.",
            "Points originally on the left are now on the right.",
            "Every point must cross the line to swap sides.",
            "This proves the windmill hits every star eventually."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # Colors (L008)
        BLUE_PT = "#0000FF"
        RED_PT = "#FF0000"
        GOLD_LINE = "#FFD700"
        GREEN_PT = "#00FF00"
        WHITE_HEX = "#FFFFFF"
        
        # === Animation for Lecture Line 1 ===
        # Show the initial balanced line with labeled sides and colored points
        
        # Visual elements following VideoCritic suggestions (Issue 32, 33, 34)
        pivot_dot = Dot(color=WHITE_HEX, radius=0.1)
        self.place_at_grid(pivot_dot, "D3", scale_factor=0.8)
        pivot_pos = pivot_dot.get_center()
        
        # Blue dots on the left (Issue 33)
        blue_dot1 = Dot(color=BLUE_PT, radius=0.08)
        blue_dot2 = Dot(color=BLUE_PT, radius=0.08)
        blue_dots = VGroup(blue_dot1, blue_dot2).arrange(DOWN, buff=0.6)
        self.place_in_area(blue_dots, "B2", "C2", scale_factor=0.7)
        
        # Red dots on the right (Issue 33)
        red_dot1 = Dot(color=RED_PT, radius=0.08)
        red_dot2 = Dot(color=RED_PT, radius=0.08)
        red_dots = VGroup(red_dot1, red_dot2).arrange(DOWN, buff=0.6)
        self.place_in_area(red_dots, "E4", "F4", scale_factor=0.7)
        
        # Labels (Issue 32)
        label_l = Text("L", font_size=24, color=BLUE_PT)
        label_r = Text("R", font_size=24, color=RED_PT)
        self.place_at_grid(label_l, "D2", scale_factor=0.8)
        self.place_at_grid(label_r, "D4", scale_factor=0.8)
        
        # Initial line (start vertical)
        line = Line(pivot_pos + DOWN*3, pivot_pos + UP*3, color=GOLD_LINE, stroke_width=4)
        
        self.lecture[0].set_color(GOLD_LINE)
        self.play(
            FadeIn(pivot_dot), 
            FadeIn(blue_dots), 
            FadeIn(red_dots), 
            Create(line), 
            Write(label_l), 
            Write(label_r)
        )
        self.wait(1.5)
        
        # === Animation for Lecture Line 2 ===
        # Rotate the line slowly 180 degrees using a 'smooth' rate function
        
        self.lecture[0].set_color(WHITE_HEX)
        self.lecture[1].set_color(GOLD_LINE)
        
        angle_tracker = ValueTracker(0)
        
        # Updaters for rotation (L010/L011)
        def line_updater(m):
            angle = PI/2 + angle_tracker.get_value()
            direction = np.array([np.cos(angle), np.sin(angle), 0])
            m.put_start_and_end_on(pivot_pos - 3.5 * direction, pivot_pos + 3.5 * direction)
            
        def labels_updater(vgroup):
            angle = PI/2 + angle_tracker.get_value()
            # Normal vector for 'Left' side relative to the line direction
            normal = np.array([-np.sin(angle), np.cos(angle), 0])
            # Using Proximity Rule (L002) - approx 1 unit distance
            vgroup[0].move_to(pivot_pos + normal * 1.0)
            vgroup[1].move_to(pivot_pos - normal * 1.0)
            
        labels_vgroup = VGroup(label_l, label_r)
        line.add_updater(line_updater)
        labels_vgroup.add_updater(labels_updater)
        
        # Perform 180-degree rotation (PI radians)
        self.play(angle_tracker.animate.set_value(PI), run_time=5, rate_func=rate_functions.smooth)
        self.wait(3.0)
        
        line.clear_updaters()
        labels_vgroup.clear_updaters()
        
        # === Animation for Lecture Line 3 ===
        # Morph the side labels and point colors to show the sides have completely swapped.
        
        self.lecture[1].set_color(WHITE_HEX)
        self.lecture[2].set_color(BLUE_PT)
        
        # Swap colors visually to represent the change in state relative to the line orientation
        self.play(
            blue_dots.animate.set_color(RED_PT),
            red_dots.animate.set_color(BLUE_PT),
            label_l.animate.set_color(RED_PT),
            label_r.animate.set_color(BLUE_PT),
            run_time=2
        )
        self.wait(2.0)
        
        # === Animation for Lecture Line 4 ===
        # Highlight each point in sequence as the line passes through it.
        
        self.lecture[2].set_color(WHITE_HEX)
        self.lecture[3].set_color(GOLD_LINE)
        
        # Indicate points to represent their crossing event
        all_stars = [blue_dot1, blue_dot2, red_dot1, red_dot2]
        for star in all_stars:
            self.play(Indicate(star, color=GOLD_LINE, scale_factor=1.5), run_time=0.8)
        
        self.wait(1.5)
        
        # === Animation for Lecture Line 5 ===
        # Flash all points in green (#00FF00) to signify the proof is complete.
        
        self.lecture[3].set_color(WHITE_HEX)
        self.lecture[4].set_color(GREEN_PT)
        
        all_points = VGroup(pivot_dot, *all_stars)
        self.play(
            *[Flash(p, color=GREEN_PT) for p in all_points],
            all_points.animate.set_color(GREEN_PT),
            run_time=2
        )
        
        self.wait(2.0)
