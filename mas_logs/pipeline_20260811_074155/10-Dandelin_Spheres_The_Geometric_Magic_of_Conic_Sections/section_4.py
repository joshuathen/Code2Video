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
        self.setup_layout("The Step-by-Step Proof", [
            "Pick any point P on the intersection curve.",
            "Connect P to both Dandelin foci, F1 and F2.",
            "PF1 and PF2 equal the distances to the circles.",
            "The sum PF1 + PF2 equals the slant distance.",
            "This constant sum defines the shape as an ellipse."
        ])

        # === Animation for Lecture Line 1 ===
        # Highlight point P (green #00FF00) on the ellipse (yellow #FFFF00).
        # Fix for Issue 20 & 21: Moved to area C3-F6 and reduced scale to 0.75
        ellipse = Ellipse(width=3.6, height=2.2, color="#FFFF00")
        self.place_in_area(ellipse, "C3", "F6", scale_factor=0.75)
        
        center = ellipse.get_center()
        a_p = 1.8 * 0.75
        b_p = 1.1 * 0.75
        c_p = np.sqrt(a_p**2 - b_p**2)
        
        t_tracker = ValueTracker(PI / 3)
        
        p_dot = Dot(color="#00FF00", z_index=10)
        p_dot.add_updater(lambda d: d.move_to(center + np.array([
            a_p * np.cos(t_tracker.get_value()), 
            b_p * np.sin(t_tracker.get_value()), 
            0
        ])))
        
        p_label = Text("P", font_size=20, color="#00FF00")
        p_label.add_updater(lambda l: l.next_to(p_dot, UR, buff=0.1))
        
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        self.play(Create(ellipse))
        self.play(FadeIn(p_dot), Write(p_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Connect P to both Dandelin foci, F1 and F2. PF1 and distance to upper circle match.
        f1_pos = center + LEFT * c_p
        f2_pos = center + RIGHT * c_p
        f1 = Dot(f1_pos, color="#FF0000")
        f2 = Dot(f2_pos, color="#FF0000")
        f1_text = Text("F1", font_size=18, color="#FF0000").next_to(f1, DOWN, buff=0.1)
        f2_text = Text("F2", font_size=18, color="#FF0000").next_to(f2, DOWN, buff=0.1)
        
        line_pf1 = Line(color="#FF0000", stroke_width=4)
        line_pf1.add_updater(lambda l: l.set_points_as_corners([f1.get_center(), p_dot.get_center()]))
        
        # Upper circle visualization (horizontal dashed line)
        # Positioned relative to the ellipse
        c1_y = center[1] + 1.8 # Fixed height for upper circle
        c1_ref = DashedLine(self.grid['A3'], self.grid['A6'], color=GRAY).set_y(c1_y)
        
        # Generator segment to upper circle
        line_p_c1 = Line(color="#00FFFF", stroke_width=4)
        line_p_c1.add_updater(lambda l: l.set_points_as_corners([
            p_dot.get_center(), 
            p_dot.get_center() + np.array([-0.2, 1, 0]) * (a_p + c_p * np.cos(t_tracker.get_value()))
        ]))
        
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        self.play(FadeIn(f1), FadeIn(f2), Write(f1_text), Write(f2_text))
        self.play(Create(line_pf1))
        self.play(Create(c1_ref), Create(line_p_c1))
        self.play(Indicate(line_pf1), Indicate(line_p_c1, color="#00FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # PF2 and distance to lower circle match.
        line_pf2 = Line(color="#FF0000", stroke_width=4)
        line_pf2.add_updater(lambda l: l.set_points_as_corners([f2.get_center(), p_dot.get_center()]))
        
        # Lower circle visualization
        c2_y = center[1] - 1.8 # Fixed height for lower circle
        c2_ref = DashedLine(self.grid['F3'], self.grid['F6'], color=GRAY).set_y(c2_y)
        
        # Generator segment to lower circle
        line_p_c2 = Line(color="#FF00FF", stroke_width=4)
        line_p_c2.add_updater(lambda l: l.set_points_as_corners([
            p_dot.get_center(), 
            p_dot.get_center() + np.array([0.2, -1, 0]) * (a_p - c_p * np.cos(t_tracker.get_value()))
        ]))
        
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.play(Create(line_pf2))
        self.play(Create(c2_ref), Create(line_p_c2))
        self.play(Indicate(line_pf2), Indicate(line_p_c2, color="#FF00FF"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight the slant segment (orange #FFA500) between the two circles.
        # This segment is the union of the two generator segments.
        slant_seg = Line(color="#FFA500", stroke_width=6, z_index=-1)
        slant_seg.add_updater(lambda l: l.set_points_as_corners([
            p_dot.get_center() + np.array([-0.2, 1, 0]) * (a_p + c_p * np.cos(t_tracker.get_value())),
            p_dot.get_center() + np.array([0.2, -1, 0]) * (a_p - c_p * np.cos(t_tracker.get_value()))
        ]))
        
        self.play(self.lecture[3].animate.set_color("#FFA500"))
        self.play(FadeIn(slant_seg))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Move P (green #00FF00) along the ellipse (yellow #FFFF00) while keeping the sum PF1+PF2 constant.
        self.play(self.lecture[4].animate.set_color("#FFFF00"))
        self.play(t_tracker.animate.set_value(2 * PI + PI / 3), run_time=8, rate_func=linear)
        self.wait(2)
