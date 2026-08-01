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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines_text = [
            'Map every pair of points to 3D space.', 
            "The xy-coordinate is the pair's midpoint.", 
            'The height represents the distance between points.', 
            'These points form a continuous surface in space.', 
            'Each surface point represents a unique chord.'
        ]
        self.setup_layout("The Geometry of Pairs: Mapping to 3D Space", lecture_lines_text)

        # Colors
        GREEN = "#00FF00"
        ORANGE = "#FFA500"
        PURPLE = "#800080"
        WHITE = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        def curve_func(t):
            # A squished heart-like blob
            r = 1.2 + 0.3 * np.sin(2 * t)
            return np.array([
                r * np.cos(t),
                0.6 * r * np.sin(t) - 1.0, 
                0
            ])

        jordan_curve = ParametricFunction(curve_func, t_range=[0, TAU], color=WHITE)
        # Resolved Issue 39: Moved from C2-F5 to D2-F5 to leave space above for 3D visualization
        self.place_in_area(jordan_curve, "D2", "F5", scale_factor=0.9)
        
        t_a = 0.5
        t_b = 2.5
        
        pos_a = jordan_curve.point_from_proportion(t_a / TAU)
        pos_b = jordan_curve.point_from_proportion(t_b / TAU)
        
        dot_a = Dot(pos_a, color=GREEN)
        dot_b = Dot(pos_b, color=GREEN)
        label_a = Text("A", font_size=16, color=GREEN).next_to(dot_a, LEFT, buff=0.1)
        label_b = Text("B", font_size=16, color=GREEN).next_to(dot_b, RIGHT, buff=0.1)
        
        self.play(Create(jordan_curve), FadeIn(dot_a, dot_b, label_a, label_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        chord = Line(pos_a, pos_b, color=GREEN, stroke_width=2)
        mid_pos = (pos_a + pos_b) / 2
        dot_m = Dot(mid_pos, color=ORANGE)
        label_m = Text("M", font_size=16, color=ORANGE).next_to(dot_m, DOWN, buff=0.1)
        
        self.play(Create(chord), FadeIn(dot_m, label_m))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        dist_ab = np.linalg.norm(pos_a - pos_b)
        height_vec = UP * dist_ab * 0.8 
        tip_pos = mid_pos + height_vec
        
        # Use DashedLine instead of Line
        height_line = DashedLine(mid_pos, tip_pos, color=ORANGE)
        dot_p = Dot(tip_pos, color=WHITE)
        label_p = Text("P(x,y,z)", font_size=16, color=WHITE).next_to(dot_p, UP, buff=0.1)
        
        self.play(Create(height_line), FadeIn(dot_p, label_p))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        move_tracker = ValueTracker(0)
        trace_p = VMobject(color=YELLOW_A, stroke_width=2)
        trace_p.set_points_as_corners([tip_pos, tip_pos])

        def update_all(mob):
            val = move_tracker.get_value()
            new_ta = (t_a + val) % TAU
            new_tb = (t_b + val * 1.5) % TAU 
            
            pa = jordan_curve.point_from_proportion(new_ta / TAU)
            pb = jordan_curve.point_from_proportion(new_tb / TAU)
            pm = (pa + pb) / 2
            
            dot_a.move_to(pa)
            dot_b.move_to(pb)
            label_a.next_to(dot_a, LEFT, buff=0.1)
            label_b.next_to(dot_b, RIGHT, buff=0.1)
            chord.put_start_and_end_on(pa, pb)
            
            dot_m.move_to(pm)
            label_m.next_to(dot_m, DOWN, buff=0.1)
            
            dist = np.linalg.norm(pa - pb)
            h_vec = UP * dist * 0.8
            new_tip = pm + h_vec
            
            height_line.put_start_and_end_on(pm, new_tip)
            dot_p.move_to(new_tip)
            label_p.next_to(dot_p, UP, buff=0.1)
            
            trace_p.add_line_to(new_tip)

        self.add(trace_p)
        dot_a.add_updater(update_all)
        self.play(move_tracker.animate.set_value(2), run_time=4, rate_func=linear)
        dot_a.remove_updater(update_all)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        surface_blob = Ellipse(width=3, height=1.5, color=PURPLE, fill_opacity=0.4)
        # Resolved Issue 38: Moved from B2-D5 to A2-C5 to avoid overlap with the lower curve
        self.place_in_area(surface_blob, "A2", "C5", scale_factor=0.8)
        
        self.play(
            FadeIn(surface_blob), 
            FadeOut(trace_p), 
            FadeOut(height_line), 
            FadeOut(label_p), 
            FadeOut(dot_p)
        )
        self.wait(2)
