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
        self.setup_layout("The Geometric Bridge", [
            "Pick any point P on the ellipse's edge.",
            "Link P to the sphere's contact point.",
            "Trace from P along the cone's surface.",
            "These two segments are equal tangent lengths.",
            "This bridge connects the plane and cone."
        ])
        
        # === Geometry Constants based on Analysis ===
        # Center: C4 (3.5, 0.2), Radius: 1.0 (Fits columns 3-5)
        # F1 (Tangent point on plane): C5 (4.5, 0.2)
        # Q1 (Tangent point on cone): D4 (3.5, -0.8)
        # P (External point): D5 (4.5, -0.8)
        
        # === Load Assets and Create Shapes ===
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg]
        sphere_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        self.place_in_area(sphere_asset, "B3", "D5", scale_factor=1.2)
        sphere_asset.set_color("#FFD700")
        
        # Represent the sphere boundary with a circle for precision
        sphere_circ = Circle(radius=1.0, color="#FFD700", stroke_width=2)
        self.place_at_grid(sphere_circ, "C4")
        
        # Tangent Point 1: F1 (Contact with slicing plane)
        f1_dot = Dot(color=RED)
        self.place_at_grid(f1_dot, "C5")
        f1_label = MathTex("F_1", color=RED, font_size=30).next_to(f1_dot, RIGHT, buff=0.1)
        
        # Slicing Plane line (tangent at F1)
        plane_line = Line(self.grid["A5"], self.grid["F5"], color=WHITE, stroke_width=1.5)
        
        # Tangent Point 2: Q1 (Contact with cone rim)
        q1_dot = Dot(color="#FFD700")
        self.place_at_grid(q1_dot, "D4")
        q1_label = MathTex("Q_1", color="#FFD700", font_size=30).next_to(q1_dot, DOWN, buff=0.1)
        
        # Cone wall line (tangent at Q1)
        cone_line = Line(self.grid["D1"], self.grid["D6"], color="#888888", stroke_width=1.5)
        
        # Point P: Where the plane and cone wall meet
        p_dot = Dot(color=WHITE)
        self.place_at_grid(p_dot, "D5")
        p_label = MathTex("P", color=WHITE, font_size=34).next_to(p_dot, DR, buff=0.1)

        # Equality labels
        eq_math = MathTex("PF_1 = PQ_1", color="#FFA500", font_size=36)
        self.place_at_grid(eq_math, "B4")

        # Initial Persistent Elements
        self.add(sphere_asset, sphere_circ, plane_line, cone_line)

        # === Animation for Lecture Line 1 ===
        # "Pick any point P on the ellipse's edge."
        self.lecture[0].set_color(WHITE)
        self.play(FadeIn(p_dot), Write(p_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Link P to the sphere's contact point."
        self.lecture[1].set_color("#FFA500")
        pf1_seg = Line(p_dot.get_center(), f1_dot.get_center(), color="#FFA500", stroke_width=5)
        self.play(FadeIn(f1_dot), Write(f1_label), Create(pf1_seg))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Trace from P along the cone's surface."
        self.lecture[2].set_color("#FFA500")
        pq1_seg = Line(p_dot.get_center(), q1_dot.get_center(), color="#FFA500", stroke_width=5)
        self.play(FadeIn(q1_dot), Write(q1_label), Create(pq1_seg))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "These two segments are equal tangent lengths."
        self.lecture[3].set_color("#FFA500")
        
        self.play(
            Write(eq_math),
            pf1_seg.animate.set_stroke(width=8),
            pq1_seg.animate.set_stroke(width=8)
        )
        self.play(
            pf1_seg.animate.set_stroke(width=5),
            pq1_seg.animate.set_stroke(width=5)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "This bridge connects the plane and cone."
        self.lecture[4].set_color(YELLOW)
        
        # Final highlight
        bridge_highlight = VGroup(pf1_seg, pq1_seg, f1_dot, q1_dot, p_dot)
        self.play(
            Indicate(bridge_highlight, color=YELLOW),
            ScaleInPlace(sphere_asset, 1.1),
            rate_func=there_and_back
        )
        self.wait(2)
