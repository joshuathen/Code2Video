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
        # Setup the layout with title and lecture lines
        self.setup_layout("Visual Walkthrough: Slopes of the Circle", [
            "For our circle, dy/dx equals negative x over y.",
            "At the top, x is zero, making the slope zero.",
            "At the side, y is zero, making the slope undefined.",
            "Every coordinate determines a unique tangent line slope.",
            "Watch the slope change as we move around."
        ])

        # === Animation for Lecture Line 1 ===
        # Highlight first line and show the formula isolation
        self.lecture[0].set_color("#00FF00")
        formula = Text("dy/dx = -x/y", color=WHITE, font_size=32)
        box = SurroundingRectangle(formula, color="#00FF00", buff=0.2)
        formula_group = VGroup(formula, box)
        # Resolved Issue 40: Corrected placement from 'A2', 'B5' to 'A4', 'B6'
        self.place_in_area(formula_group, 'A4', 'B6', scale_factor=0.8)
        
        self.play(Create(box), Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line and show circle with horizontal tangent
        self.lecture[1].set_color("#FFFFFF")
        
        # Draw a blue circle (#1E90FF)
        circle = Circle(radius=1.3, color="#1E90FF")
        # Resolved Issue 41: Corrected placement from 'C2', 'F5' to 'C3', 'F6'
        self.place_in_area(circle, 'C3', 'F6')
        center = circle.get_center()
        
        # Point at the top (scaled to radius r)
        r = 1.3
        pos_top = center + UP * r
        point = Dot(pos_top, color=WHITE)
        
        # Horizontal tangent line at the top
        tangent = Line(pos_top + LEFT * 0.9, pos_top + RIGHT * 0.9, color="#FFFFFF")
        
        self.play(Create(circle))
        self.play(FadeIn(point), Create(tangent))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third line and show vertical tangent at the side
        self.lecture[2].set_color("#FF4500")
        
        # Move the point to the side (scaled to radius r)
        pos_side = center + RIGHT * r
        # Vertical line for undefined slope
        new_tangent = Line(pos_side + UP * 0.9, pos_side + DOWN * 0.9, color="#FF4500")
        
        self.play(
            point.animate.move_to(pos_side),
            ReplacementTransform(tangent, new_tangent)
        )
        tangent = new_tangent
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight fourth line and show point (3, 4) with slope label
        self.lecture[3].set_color("#1E90FF")
        
        # Point at proportional coordinates (3,4) corresponds to angle atan2(4, 3)
        angle_34 = np.arctan2(4, 3)
        pos_34 = center + np.array([np.cos(angle_34), np.sin(angle_34), 0]) * r
        
        # Tangent vector is (-sin(theta), cos(theta))
        t_vec = np.array([-np.sin(angle_34), np.cos(angle_34), 0])
        new_tangent = Line(pos_34 - t_vec * 0.9, pos_34 + t_vec * 0.9, color="#FFFF00")
        
        slope_label = Text("slope = -3/4", font_size=20, color="#FFFFFF")
        # Resolved Issue 42: Corrected placement from 'C5' to 'B3'
        self.place_at_grid(slope_label, 'B3', scale_factor=0.8)
        
        self.play(
            point.animate.move_to(pos_34),
            ReplacementTransform(tangent, new_tangent),
            FadeIn(slope_label)
        )
        tangent = new_tangent
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight fifth line and animate rotation around circle
        self.lecture[4].set_color("#FFFF00")
        self.play(FadeOut(slope_label))
        
        # Use a ValueTracker for continuous circular movement
        rot_tracker = ValueTracker(angle_34)
        
        def update_point_pos(m):
            theta = rot_tracker.get_value()
            m.move_to(center + np.array([np.cos(theta), np.sin(theta), 0]) * r)
            
        def update_tangent_line(m):
            p = point.get_center()
            theta = rot_tracker.get_value()
            tv = np.array([-np.sin(theta), np.cos(theta), 0])
            m.put_start_and_end_on(p - tv * 0.9, p + tv * 0.9)
            m.set_color("#FFFF00")

        point.add_updater(update_point_pos)
        tangent.add_updater(update_tangent_line)
        
        # Perform full rotation
        self.play(rot_tracker.animate.set_value(angle_34 + 2 * PI), run_time=5, rate_func=linear)
        
        point.remove_updater(update_point_pos)
        tangent.remove_updater(update_tangent_line)
        self.wait(2)
