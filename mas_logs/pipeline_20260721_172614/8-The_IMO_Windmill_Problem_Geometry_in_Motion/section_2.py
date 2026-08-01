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
        # Setup
        title = "Prerequisite: Discrete vs. Continuous Motion"
        lines = [
            "The beam's rotation angle changes continuously over time.",
            "But the center of rotation jumps at discrete hits.",
            "No three points are collinear to keep pivots unique."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_PIVOT = "#2ECC71"  # Green
        COLOR_LINE = YELLOW
        COLOR_POINT = WHITE
        COLOR_X = "#E74C3C"      # Red

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Load asset - Issue 21
        beam = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/beam.svg")
        beam.set_color(COLOR_LINE)
        # Ensure it looks like a long beam
        beam.stretch_to_fit_width(4)
        beam.stretch_to_fit_height(0.05)
        
        # Points
        dot1 = Dot(self.grid["B3"], color=COLOR_PIVOT, radius=0.1)
        dot2 = Dot(self.grid["D4"], color=COLOR_POINT, radius=0.1)
        label1 = Text("P1", font_size=16).next_to(dot1, UP, buff=0.1)
        label2 = Text("P2", font_size=16).next_to(dot2, DOWN, buff=0.1)
        
        pivot_system = VGroup(dot1, dot2, label1, label2)
        # Issue 27 fix: Positioning of the pivot system
        self.place_in_area(pivot_system, 'A3', 'F6', scale_factor=0.7)
        
        # Get updated positions for beam rotation logic after placement
        p1_curr = dot1.get_center()
        p2_curr = dot2.get_center()
        
        # Angle tracker and updater
        angle_tracker = ValueTracker(0)
        current_pivot = [p1_curr]
        
        # Use an attribute to track last angle for incremental rotation
        beam.last_angle = 0
        def beam_updater(m):
            m.move_to(current_pivot[0])
            curr_angle = angle_tracker.get_value()
            m.rotate(curr_angle - m.last_angle)
            m.last_angle = curr_angle
            
        beam.add_updater(beam_updater)
        
        self.add(pivot_system, beam)
        self.play(angle_tracker.animate.set_value(PI/2), run_time=2, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Calculate target angle to hit P2
        target_vec = p2_curr - p1_curr
        target_angle = np.arctan2(target_vec[1], target_vec[0])
        
        # Rotate to the hit
        self.play(angle_tracker.animate.set_value(target_angle), run_time=1.5, rate_func=linear)
        
        # Pivot Flash and Jump
        flash = Circle(radius=0.3, color=COLOR_PIVOT, stroke_width=4).move_to(p2_curr)
        self.play(
            dot2.animate.set_color(COLOR_PIVOT),
            dot1.animate.set_color(COLOR_POINT),
            FadeIn(flash, scale=0.5),
            run_time=0.3
        )
        self.play(FadeOut(flash), run_time=0.2)
        
        # Jump pivot point
        current_pivot[0] = p2_curr
        
        # Continue rotation
        self.play(angle_tracker.animate.increment_value(PI/3), run_time=1.5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Cleanup windmill
        self.play(FadeOut(pivot_system, beam), run_time=0.5)

        # Collinearity demonstration
        cp1 = Dot(self.grid["C2"], color=COLOR_POINT)
        cp2 = Dot(self.grid["C4"], color=COLOR_POINT)
        cp3 = Dot(self.grid["C6"], color=COLOR_POINT)
        d_line = DashedLine(
            start=self.grid["C2"], 
            end=self.grid["C6"], 
            color=WHITE, 
            dash_length=0.1
        )
        
        collinear_group = VGroup(cp1, cp2, cp3, d_line)
        # Issue 28 fix: Center the collinearity group
        self.place_in_area(collinear_group, 'C2', 'C6', scale_factor=0.8)
        
        # Red X Mark - Issue 29 fix
        red_x_mark = VGroup(
            Line(LEFT, RIGHT, stroke_width=8),
            Line(UP, DOWN, stroke_width=8)
        ).rotate(45*DEGREES).set_color(COLOR_X)
        
        # Positioning and scaling per Issue 29
        self.place_at_grid(red_x_mark, 'C4', scale_factor=1.5)

        self.play(Create(collinear_group))
        self.wait(0.5)
        self.play(Create(red_x_mark))
        self.wait(2)

        # Final cleanup
        self.lecture[2].set_color(WHITE)
        self.wait(1)
