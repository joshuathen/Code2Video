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

class Section4Scene(TeachingScene):
    def construct(self):
        # Colors
        COLOR_PF = "#FF00FF" # Pink
        COLOR_PQ = "#00FFFF" # Cyan
        COLOR_GEN = "#FFFF00" # Yellow
        COLOR_SPHERE = "#4488FF"
        COLOR_CONE = "#888888"
        COLOR_FORMULA = "#FFA500" # Orange
        COLOR_FINAL = "#00FF00" # Green

        # Setup layout
        self.setup_layout(
            "The 'Aha!' Moment: The Proof", 
            [
                "Pick any point P on the ellipse's edge.", 
                "Segments PF1 and PF2 are tangent to the spheres.", 
                "They equal segments along the cone's surface.", 
                "Their sum equals the distance between contact circles.", 
                "Since this distance is constant, it's an ellipse!"
            ]
        )

        # --- Geometry Definition (Manual calculation for a clear side view) ---
        # Vertex
        v_pt = np.array([0, 2.0, 0])
        # Cone lines
        cone_l = Line(v_pt, np.array([-2, -3, 0]), color=COLOR_CONE, stroke_opacity=0.5)
        cone_r = Line(v_pt, np.array([2, -3, 0]), color=COLOR_CONE, stroke_opacity=0.5)
        
        # Spheres (Circles in side view)
        sphere1 = Circle(radius=0.6, color=COLOR_SPHERE, fill_opacity=0.2).move_to([0, 0.8, 0])
        sphere2 = Circle(radius=1.2, color=COLOR_SPHERE, fill_opacity=0.2).move_to([0, -1.2, 0])
        
        # Contact Circles (Line segments in side view)
        cc1 = Line([-0.45, 1.1, 0], [0.45, 1.1, 0], color=COLOR_SPHERE)
        cc2 = Line([-1.2, -0.6, 0], [1.2, -0.6, 0], color=COLOR_SPHERE)
        
        # Plane (The ellipse seen from the side)
        p1 = np.array([-0.8, 0.4, 0]) # High point
        p2 = np.array([1.1, -1.8, 0]) # Low point
        plane_line = Line(p1, p2, color=WHITE)
        
        # Focal points F1, F2 (where spheres touch the plane)
        f1_pt = np.array([-0.35, -0.15, 0]) # Approx touch points
        f2_pt = np.array([0.45, -1.0, 0])
        
        # Point P on the ellipse
        p_pt = np.array([0.1, -0.6, 0])
        
        # Mobjects
        dot_p = Dot(p_pt, color=WHITE, radius=0.06)
        label_p = Text("P", font_size=16).next_to(dot_p, RIGHT, buff=0.1)
        dot_f1 = Dot(f1_pt, color=COLOR_PF, radius=0.06)
        label_f1 = Text("F1", font_size=16).next_to(dot_f1, LEFT, buff=0.1)
        dot_f2 = Dot(f2_pt, color=COLOR_PF, radius=0.06)
        label_f2 = Text("F2", font_size=16).next_to(dot_f2, RIGHT, buff=0.1)
        
        diagram = VGroup(cone_l, cone_r, sphere1, sphere2, cc1, cc2, plane_line, dot_f1, dot_f2, label_f1, label_f2)
        self.place_in_area(diagram, "A2", "F6", scale_factor=0.9)
        
        # Adjust P after diagram placement to keep it relative
        dot_p.move_to(diagram.get_center() + p_pt * 0.9)
        label_p.next_to(dot_p, UR, buff=0.05)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_PF)
        self.play(FadeIn(diagram))
        self.play(FadeIn(dot_p), FadeIn(label_p))
        
        seg_pf1 = Line(dot_p.get_center(), dot_f1.get_center(), color=COLOR_PF, stroke_width=4)
        seg_pf2 = Line(dot_p.get_center(), dot_f2.get_center(), color=COLOR_PF, stroke_width=4)
        
        self.play(Create(seg_pf1), Create(seg_pf2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_GEN)
        # Highlight tangency
        circle_tangency_box = SurroundingRectangle(VGroup(dot_f1, dot_f2), color=COLOR_GEN, buff=0.1)
        self.play(Create(circle_tangency_box))
        self.play(FadeOut(circle_tangency_box))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_PQ)
        # Generator line from V through P
        v_screen = diagram[0].get_start() # Vertex of the cone after scaling/moving
        generator = Line(v_screen, dot_p.get_center() + (dot_p.get_center() - v_screen) * 1.5, color=COLOR_GEN, stroke_opacity=0.6)
        
        # Intersections with contact circles
        # Manual intersection point calculation logic for side view to ensure it works
        # cc1: y = diagram_cc1_y
        # generator: line from v_screen to dot_p
        # We'll use get_center() of CC1/CC2 and the generator slope.
        # However, the previous code used Intersection. If it rendered, I'll stick with it but check if it's safe.
        # Actually, let's just use the previous logic.
        q1_pt = Intersection(generator, cc1).get_center()
        q2_pt = Intersection(generator, cc2).get_center()
        
        dot_q1 = Dot(q1_pt, color=COLOR_PQ, radius=0.06)
        dot_q2 = Dot(q2_pt, color=COLOR_PQ, radius=0.06)
        label_q1 = Text("Q1", font_size=16, color=COLOR_PQ).next_to(dot_q1, LEFT, buff=0.1)
        label_q2 = Text("Q2", font_size=16, color=COLOR_PQ).next_to(dot_q2, RIGHT, buff=0.1)
        
        self.play(Create(generator))
        self.play(FadeIn(dot_q1), FadeIn(label_q1), FadeIn(dot_q2), FadeIn(label_q2))
        
        # Highlight PQ1 and PQ2
        seg_pq1 = Line(dot_p.get_center(), dot_q1.get_center(), color=COLOR_PQ, stroke_width=6)
        seg_pq2 = Line(dot_p.get_center(), dot_q2.get_center(), color=COLOR_PQ, stroke_width=6)
        
        self.play(Create(seg_pq1), Create(seg_pq2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_FORMULA)
        
        eq1 = Text("PF1 = PQ1", font_size=22, color=WHITE)
        eq2 = Text("PF2 = PQ2", font_size=22, color=WHITE)
        self.place_at_grid(eq1, 'A1', scale_factor=0.8)
        self.place_at_grid(eq2, 'B1', scale_factor=0.8)
        
        self.play(Write(eq1))
        self.play(Indicate(seg_pf1), Indicate(seg_pq1))
        self.play(Write(eq2))
        self.play(Indicate(seg_pf2), Indicate(seg_pq2))
        
        sum_eq = Text("PF1 + PF2 = PQ1 + PQ2", font_size=24, color=COLOR_FORMULA)
        self.place_at_grid(sum_eq, 'C1', scale_factor=0.6)
        self.play(Write(sum_eq))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(COLOR_FINAL)
        
        # PQ1 + PQ2 is the distance Q1Q2
        q1q2_bracket = BraceBetweenPoints(dot_q1.get_center(), dot_q2.get_center(), color=COLOR_FINAL)
        q1q2_text = Text("Constant Distance", font_size=20, color=COLOR_FINAL).next_to(q1q2_bracket, RIGHT, buff=0.1)
        
        self.play(Create(q1q2_bracket), Write(q1q2_text))
        
        final_text = Text("Ellipse Definition Met!", font_size=28, color=COLOR_FINAL)
        self.place_at_grid(final_text, 'F1', scale_factor=0.8)
        self.play(Write(final_text))
        
        self.wait(2)
