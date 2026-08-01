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
        # Setup
        title_str = "The Constant Sum"
        lines_str = [
            "The sum PF1 + PF2 equals the total slant distance.",
            "This distance AB connects the two fixed circles.",
            "The length AB remains constant as point P moves.",
            "Thus, the sum of distances to foci is constant.",
            "The Dandelin spheres prove this curve is an ellipse."
        ]
        self.setup_layout(title_str, lines_str)

        # Colors
        COLOR_PF1 = "#0000FF" # Blue
        COLOR_PF2 = "#FF4500" # Orange-red
        COLOR_AB = "#00FF00"  # Green
        COLOR_TEXT = "#FFFFFF"
        HIGHLIGHT_COLOR = "#FFFF00" # Yellow

        # Assets
        sphere_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg"

        # Points
        f1_pos = self.grid["D2"] + RIGHT * 0.2
        f2_pos = self.grid["D5"] + LEFT * 0.2
        a_pos = self.grid["C3"]
        b_pos = self.grid["E5"]
        
        # Fixed distance AB length
        ab_len = np.linalg.norm(b_pos - a_pos)
        
        # P's path - Ellipse logic
        # Sum of distances to foci = constant = ab_len
        center_ellipse = (f1_pos + f2_pos) / 2
        c_dist = np.linalg.norm(f1_pos - center_ellipse)
        a_semi = ab_len / 2
        b_semi = np.sqrt(max(a_semi**2 - c_dist**2, 0.1)) # semi-minor axis
        
        ellipse_path = Ellipse(width=2*a_semi, height=2*b_semi, color=GRAY, stroke_opacity=0.3).move_to(center_ellipse)
        
        # Initial P (proportion 0.25 for a good starting angle)
        dot_p = Dot(ellipse_path.point_from_proportion(0.25), color=WHITE)
        label_p = Text("P", font_size=18).next_to(dot_p, UP, buff=0.1)
        
        dot_f1 = Dot(f1_pos, color=COLOR_PF1)
        label_f1 = Text("F1", font_size=18, color=COLOR_PF1).next_to(dot_f1, DOWN, buff=0.1)
        
        dot_f2 = Dot(f2_pos, color=COLOR_PF2)
        label_f2 = Text("F2", font_size=18, color=COLOR_PF2).next_to(dot_f2, DOWN, buff=0.1)
        
        dot_a = Dot(a_pos, color=COLOR_PF1)
        label_a = Text("A", font_size=18, color=COLOR_PF1).next_to(dot_a, LEFT, buff=0.1)
        
        dot_b = Dot(b_pos, color=COLOR_PF2)
        label_b = Text("B", font_size=18, color=COLOR_PF2).next_to(dot_b, RIGHT, buff=0.1)

        # Spheres at A and B
        sphere_a = SVGMobject(sphere_asset_path).scale(0.4).move_to(a_pos + UP*0.4)
        sphere_b = SVGMobject(sphere_asset_path).scale(0.6).move_to(b_pos + DOWN*0.6)

        # Segments
        line_pf1 = Line(dot_p.get_center(), f1_pos, color=COLOR_PF1)
        line_pf2 = Line(dot_p.get_center(), f2_pos, color=COLOR_PF2)
        
        # Components of AB segment
        dir_ab = (b_pos - a_pos) / ab_len
        
        # Formula using Text to avoid LaTeX issues
        formula = Text("PF1 + PF2 = AB", font_size=24, color=COLOR_TEXT)
        self.place_in_area(formula, 'A1', 'A6', scale_factor=1.2) # Fixed per Issue 38

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        self.add(dot_p, label_p, dot_f1, label_f1, dot_f2, label_f2, dot_a, label_a, dot_b, label_b)
        self.play(Create(line_pf1), Create(line_pf2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Show spheres and slant segment AB
        line_ab = Line(a_pos, b_pos, color=COLOR_AB, stroke_width=4)
        self.play(FadeIn(sphere_a), FadeIn(sphere_b))
        self.play(Create(line_ab))
        
        # Calculate target segments on AB for current P
        p_curr = dot_p.get_center()
        d1 = np.linalg.norm(p_curr - f1_pos)
        d2 = np.linalg.norm(p_curr - f2_pos)
        
        line_pa = Line(a_pos, a_pos + dir_ab * d1, color=COLOR_PF1, stroke_width=8)
        line_pb = Line(b_pos, b_pos - dir_ab * d2, color=COLOR_PF2, stroke_width=8)
        
        self.play(
            ReplacementTransform(line_pf1.copy(), line_pa),
            ReplacementTransform(line_pf2.copy(), line_pb),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Move P and show AB is constant
        # We need updaters for the segments
        p_tracker = ValueTracker(0.25)

        def update_p(m):
            m.move_to(ellipse_path.point_from_proportion(p_tracker.get_value() % 1.0))
        def update_label_p(m):
            m.next_to(dot_p, UP, buff=0.1)
        def update_pf1(m):
            m.set_points_as_corners([dot_p.get_center(), f1_pos])
        def update_pf2(m):
            m.set_points_as_corners([dot_p.get_center(), f2_pos])
        def update_pa(m):
            dist = np.linalg.norm(dot_p.get_center() - f1_pos)
            m.set_points_as_corners([a_pos, a_pos + dir_ab * dist])
        def update_pb(m):
            dist = np.linalg.norm(dot_p.get_center() - f2_pos)
            m.set_points_as_corners([b_pos, b_pos - dir_ab * dist])

        dot_p.add_updater(update_p)
        label_p.add_updater(update_label_p)
        line_pf1.add_updater(update_pf1)
        line_pf2.add_updater(update_pf2)
        line_pa.add_updater(update_pa)
        line_pb.add_updater(update_pb)

        self.play(p_tracker.animate.set_value(0.75), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        self.play(Create(ellipse_path))
        self.play(p_tracker.animate.set_value(1.25), run_time=3, rate_func=linear)
        
        # Cleanup
        dot_p.clear_updaters()
        label_p.clear_updaters()
        line_pf1.clear_updaters()
        line_pf2.clear_updaters()
        line_pa.clear_updaters()
        line_pb.clear_updaters()
        
        self.lecture[4].set_color(WHITE)
        self.wait(2)
