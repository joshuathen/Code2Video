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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        title_text = "The Visual Proof (The String Property)"
        lecture_lines = [
            "Segment PF1 equals the distance to the top circle.",
            "Segment PF2 equals the distance to the bottom circle.",
            "Both distances lie along the same line on the cone.",
            "Their sum is the fixed distance between the two circles.",
            "This sum remains constant, proving it is an ellipse."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Assets ===
        # Load assets once as per instructions
        sphere_icon_top = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg").set_color(BLUE_B)
        sphere_icon_bottom = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg").set_color(BLUE_D)
        cone_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg").set_color(GRAY_A).set_opacity(0.2)

        # === Base Geometry Construction ===
        # Local space construction
        v_pt_local = np.array([0, 2.8, 0])
        cone_l_local = Line(v_pt_local, [-2.5, -2.5, 0], color=GRAY_A, stroke_width=2)
        cone_r_local = Line(v_pt_local, [2.5, -2.5, 0], color=GRAY_A, stroke_width=2)
        
        s1_center_local = np.array([0, 1.2, 0])
        s1_radius = 0.7
        s1_local = Circle(radius=s1_radius, color=BLUE_B, stroke_width=3).move_to(s1_center_local)
        sphere_icon_top.scale(s1_radius * 0.8).move_to(s1_center_local)
        
        s2_center_local = np.array([0, -1.0, 0])
        s2_radius = 1.4
        s2_local = Circle(radius=s2_radius, color=BLUE_D, stroke_width=3).move_to(s2_center_local)
        sphere_icon_bottom.scale(s2_radius * 0.8).move_to(s2_center_local)

        cone_icon.scale(2.5).move_to([0, 0, 0])
        
        # Tangency angle for the ellipse cross-section line
        angle = 20 * DEGREES
        f1_pt_local = s1_center_local + s1_radius * np.array([np.sin(angle), -np.cos(angle), 0])
        f2_pt_local = s2_center_local + s2_radius * np.array([-np.sin(angle), np.cos(angle), 0])
        ellipse_line_local = Line(f1_pt_local + 1.2*LEFT, f2_pt_local + 1.2*RIGHT, color=WHITE, stroke_width=2)
        
        # Foci dots and labels
        f1_local = Dot(f1_pt_local, color=YELLOW, radius=0.06)
        f2_local = Dot(f2_pt_local, color=YELLOW, radius=0.06)
        f1_label_local = MathTex("F_1", font_size=20, color=YELLOW).next_to(f1_local, UR, buff=0.05)
        f2_label_local = MathTex("F_2", font_size=20, color=YELLOW).next_to(f2_local, DL, buff=0.05)
        
        # Contact lines where spheres touch the cone surface
        y_c1_local = 1.2 - 0.7 * 0.42 
        y_c2_local = -1.0 - 1.4 * 0.42
        c_line1_local = Line([-0.8, y_c1_local, 0], [0.8, y_c1_local, 0], color=BLUE_A, stroke_width=1.5, stroke_opacity=0.4)
        c_line2_local = Line([-1.8, y_c2_local, 0], [1.8, y_c2_local, 0], color=BLUE_C, stroke_width=1.5, stroke_opacity=0.4)

        diagram = VGroup(
            cone_icon, cone_l_local, cone_r_local, 
            s1_local, sphere_icon_top, s2_local, sphere_icon_bottom,
            ellipse_line_local, f1_local, f2_local, 
            f1_label_local, f2_label_local, c_line1_local, c_line2_local
        )
        # Resolved Issue 38: Move diagram to C3-F6 area
        self.place_in_area(diagram, "C3", "F6", scale_factor=0.7)

        # Scene coordinate references for updaters
        v_scene = cone_l_local.get_start()
        f1_scene = f1_local.get_center()
        f2_scene = f2_local.get_center()
        y_c1_scene = c_line1_local.get_center()[1]
        y_c2_scene = c_line2_local.get_center()[1]

        # === Dynamic Animation Setup ===
        p_tracker = ValueTracker(0.2)
        
        def get_p_pos():
            return ellipse_line_local.point_from_proportion(p_tracker.get_value())

        p_dot = Dot(color=WHITE, radius=0.08)
        p_dot.add_updater(lambda d: d.move_to(get_p_pos()))
        p_label = MathTex("P", font_size=24, color=WHITE)
        p_label.add_updater(lambda l: l.next_to(p_dot, UP, buff=0.1))

        # Function to find intersection of generator line (VP) with horizontal contact lines
        def get_t_pos(y_target):
            v = v_scene
            p = get_p_pos()
            if abs(p[1] - v[1]) < 0.001: return v
            t = (y_target - v[1]) / (p[1] - v[1])
            return v + t * (p - v)

        t1_dot = Dot(color=BLUE_A, radius=0.05)
        t1_dot.add_updater(lambda d: d.move_to(get_t_pos(y_c1_scene)))
        t2_dot = Dot(color=BLUE_C, radius=0.05)
        t2_dot.add_updater(lambda d: d.move_to(get_t_pos(y_c2_scene)))
        
        t1_label = MathTex("T_1", font_size=18, color=BLUE_A)
        t1_label.add_updater(lambda l: l.next_to(t1_dot, RIGHT, buff=0.05))
        t2_label = MathTex("T_2", font_size=18, color=BLUE_C)
        t2_label.add_updater(lambda l: l.next_to(t2_dot, RIGHT, buff=0.05))

        # Dynamic tangent segments - using add_updater for performance as per instructions
        pf1_seg = Line(f1_scene, f1_scene, color="#FF8C00", stroke_width=4)
        pf1_seg.add_updater(lambda l: l.put_start_and_end_on(get_p_pos(), f1_scene))
        
        pt1_seg = Line(f1_scene, f1_scene, color="#FF8C00", stroke_width=4)
        pt1_seg.add_updater(lambda l: l.put_start_and_end_on(get_p_pos(), t1_dot.get_center()))
        
        pf2_seg = Line(f2_scene, f2_scene, color="#00BFFF", stroke_width=4)
        pf2_seg.add_updater(lambda l: l.put_start_and_end_on(get_p_pos(), f2_scene))
        
        pt2_seg = Line(f2_scene, f2_scene, color="#00BFFF", stroke_width=4)
        pt2_seg.add_updater(lambda l: l.put_start_and_end_on(get_p_pos(), t2_dot.get_center()))

        generator = Line(v_scene, v_scene, color=GRAY_B, stroke_width=1, stroke_opacity=0.6)
        generator.add_updater(lambda l: l.put_start_and_end_on(v_scene, get_p_pos()))
        
        t1t2_seg = Line(f1_scene, f1_scene, color="#FFD700", stroke_width=5)
        t1t2_seg.add_updater(lambda l: l.put_start_and_end_on(t1_dot.get_center(), t2_dot.get_center()))

        # --- Animations ---

        # === Animation for Lecture Line 1 ===
        # Highlight segments PF1 and PT1, demonstrating they are equal tangents to the top sphere
        self.lecture[0].set_color("#FF8C00")
        self.add(diagram, p_dot, p_label)
        self.play(Create(pf1_seg), Create(pt1_seg), FadeIn(t1_dot, t1_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight segments PF2 and PT2, demonstrating they are equal tangents to the bottom sphere
        self.lecture[1].set_color("#00BFFF")
        self.play(Create(pf2_seg), Create(pt2_seg), FadeIn(t2_dot, t2_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show that PF1 + PF2 is equal to the length of segment T1T2 on the generator line of the cone
        self.lecture[2].set_color(WHITE)
        self.play(Create(generator))
        self.play(Create(t1t2_seg))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Their sum is the fixed distance between the two circles.
        self.lecture[3].set_color("#FFD700")
        formula = MathTex("PF_1 + PF_2 = T_1T_2", font_size=28, color="#FFD700")
        # Resolved Issue 39: Move formula to A4-A6 area
        self.place_in_area(formula, "A4", "A6", scale_factor=0.8)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This sum remains constant, proving it is an ellipse.
        self.lecture[4].set_color("#FFD700")
        # Animate P moving along the ellipse cross-section
        self.play(p_tracker.animate.set_value(0.8), run_time=3, rate_func=linear)
        self.play(p_tracker.animate.set_value(0.1), run_time=3, rate_func=linear)
        
        final_text = Text("PF1 + PF2 = Constant", font_size=24, color="#FFD700")
        # Resolved Issue 40: Move final_text to B4-B6 area
        self.place_in_area(final_text, "B4", "B6", scale_factor=0.8)
        self.play(FadeIn(final_text))
        self.wait(2)
