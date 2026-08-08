from manim import *

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
        title = "Prerequisite: The String & The Sphere"
        lines = [
            "Ellipses have two foci with constant distance sum.",
            "Tangents from one point to a sphere are equal.",
            "These rules help unlock the cone's secret."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Colors
        color_ellipse = "#00FFFF"  # Cyan
        color_foci = "#FF0000"    # Red
        color_lines = "#FFFF00"   # Yellow
        
        self.lecture[0].set_color(color_ellipse)
        
        # Ellipse
        # Using a=2, b=1.25 -> c = 1.56
        ellipse = Ellipse(width=4.0, height=2.5, color=color_ellipse)
        self.place_in_area(ellipse, 'B2', 'E6')
        center_pos = ellipse.get_center()
        c = 1.56
        
        focus1_pos = center_pos + LEFT * c
        focus2_pos = center_pos + RIGHT * c
        
        focus1 = Dot(focus1_pos, color=color_foci)
        focus2 = Dot(focus2_pos, color=color_foci)
        
        # Top Label (Issue 29: Span A2-A6)
        foci_label = Text("Foci Locations", font_size=24, color=color_foci)
        self.place_in_area(foci_label, 'A2', 'A6')
        
        self.play(Create(ellipse))
        self.play(FadeIn(focus1, focus2), Write(foci_label))
        
        # Constant sum demonstration
        angle_tracker = ValueTracker(PI/4)
        
        def get_point_on_ellipse():
            theta = angle_tracker.get_value()
            return center_pos + np.array([2 * np.cos(theta), 1.25 * np.sin(theta), 0])
        
        moving_point = Dot(get_point_on_ellipse(), color=WHITE)
        moving_point.add_updater(lambda m: m.move_to(get_point_on_ellipse()))
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/string.svg]
        string_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/string.svg")
        string_icon.scale(0.3)
        string_icon.add_updater(lambda m: m.move_to(get_point_on_ellipse() + UP * 0.3))
        
        line1 = Line(focus1_pos, get_point_on_ellipse(), color=color_lines)
        line2 = Line(focus2_pos, get_point_on_ellipse(), color=color_lines)
        
        line1.add_updater(lambda l: l.set_points_as_corners([focus1_pos, get_point_on_ellipse()]))
        line2.add_updater(lambda l: l.set_points_as_corners([focus2_pos, get_point_on_ellipse()]))
        
        # B002: Math formula in footer area (F2-F6)
        sum_text = MathTex("d_1 + d_2 = \\text{constant}", font_size=32, color=color_lines)
        self.place_in_area(sum_text, 'F2', 'F6')
        
        self.play(FadeIn(moving_point, line1, line2, string_icon), Write(sum_text))
        self.play(angle_tracker.animate.set_value(PI/4 + 2*PI), run_time=4, rate_func=linear)
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        color_sphere = "#0000FF"   # Blue
        color_tangent = "#00FF00"  # Green
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color_sphere),
            FadeOut(ellipse, focus1, focus2, foci_label, moving_point, line1, line2, sum_text, string_icon)
        )
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg]
        # Issue 27: Consistent scale (B2-E6)
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        sphere.set_color(color_sphere)
        self.place_in_area(sphere, 'B2', 'E6')
        sphere_center = sphere.get_center()
        
        # Radius for tangent math (estimated based on grid span)
        r_val = 1.5
        
        # External Point (Issue 21: Column 6 to avoid lecture)
        p_pos = self.grid['C6']
        point_p = Dot(p_pos, color=WHITE)
        
        # Calculate Tangent Points
        d_val = np.linalg.norm(p_pos - sphere_center)
        local_p = p_pos - sphere_center
        angle_p = np.arctan2(local_p[1], local_p[0])
        alpha = np.arccos(r_val / d_val)
        
        t1_pos = sphere_center + np.array([r_val * np.cos(angle_p + alpha), r_val * np.sin(angle_p + alpha), 0])
        t2_pos = sphere_center + np.array([r_val * np.cos(angle_p - alpha), r_val * np.sin(angle_p - alpha), 0])
        
        tangent1 = Line(p_pos, t1_pos, color=color_tangent)
        tangent2 = Line(p_pos, t2_pos, color=color_tangent)
        
        # B012: Labels near mobjects
        l1_label = MathTex("L_1", font_size=28, color=color_tangent).next_to(tangent1, UP, buff=0.1)
        l2_label = MathTex("L_2", font_size=28, color=color_tangent).next_to(tangent2, DOWN, buff=0.1)
        
        # Formula (Issue 28: Consistent footer F2-F6)
        equal_text = MathTex("L_1 = L_2", font_size=32, color=color_tangent)
        self.place_in_area(equal_text, 'F2', 'F6')
        
        # Label (Issue 29: Span A2-A6)
        sphere_label = Text("3D Sphere Model", font_size=24, color=color_sphere)
        self.place_in_area(sphere_label, 'A2', 'A6')

        self.play(Create(sphere), Write(sphere_label))
        self.play(FadeIn(point_p))
        self.play(Create(tangent1), Create(tangent2))
        self.play(Write(l1_label), Write(l2_label))
        self.play(Write(equal_text))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        self.wait(2)
        
        # Final cleanup
        self.play(FadeOut(sphere, sphere_label, point_p, tangent1, tangent2, l1_label, l2_label, equal_text))
        self.wait(1)
