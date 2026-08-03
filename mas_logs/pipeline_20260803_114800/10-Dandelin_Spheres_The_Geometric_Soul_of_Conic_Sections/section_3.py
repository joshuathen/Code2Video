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
        self.setup_layout("Enter the Dandelin Spheres", [
            "Place two spheres inside the cone for an ellipse.",
            "Each sphere touches the cone along a perfect circle.",
            "The spheres also touch the cutting plane at one point.",
            "Germinal Dandelin introduced these 'kissing spheres' in 1822.",
            "These spheres reveal the hidden geometry of the conic slice."
        ])

        # Colors
        COLOR_CONE = "#C0C0C0"
        COLOR_PLANE = "#00FFFF"
        COLOR_SPHERE_UP = "#FF69B4"
        COLOR_SPHERE_DOWN = "#1E90FF"
        COLOR_CONTACT = "#ADFF2F"
        COLOR_FOCAL = "#FFA500"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_SPHERE_UP)
        
        # Cone Asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg]
        cone_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg").set_color(COLOR_CONE).scale(1.5)
        
        # Cutting plane
        plane_line = Line(LEFT * 1.5 + DOWN * 0.5, RIGHT * 1.5 + UP * 1.5, color=COLOR_PLANE)
        
        # Sphere Assets [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg]
        sphere_up = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg").set_color(COLOR_SPHERE_UP).set_opacity(0.6).scale(0.3).move_to(UP * 1.0 + LEFT * 0.1)
        sphere_down = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg").set_color(COLOR_SPHERE_DOWN).set_opacity(0.6).scale(0.8).move_to(DOWN * 1.2 + RIGHT * 0.1)
        
        geom_group = VGroup(cone_svg, plane_line, sphere_up, sphere_down)
        # Resolved Issue 33: Adjusted placement to avoid crowding
        self.place_in_area(geom_group, "A2", "E6", scale_factor=0.8)
        
        self.play(Create(cone_svg))
        self.play(Create(plane_line))
        self.play(FadeIn(sphere_up), FadeIn(sphere_down))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_CONTACT)
        
        # Tangency circles (visualized as lines in 2D projection)
        # Positioned relative to the SVG spheres
        c_line1 = Line(
            sphere_up.get_center() + LEFT * 0.3 + UP * 0.1,
            sphere_up.get_center() + RIGHT * 0.3 + UP * 0.1,
            color=COLOR_CONTACT, stroke_width=4
        )
        c_line2 = Line(
            sphere_down.get_center() + LEFT * 0.8 + DOWN * 0.2,
            sphere_down.get_center() + RIGHT * 0.8 + DOWN * 0.2,
            color=COLOR_CONTACT, stroke_width=4
        )
        
        self.play(Create(c_line1), Create(c_line2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_FOCAL)
        
        # Focal points where spheres touch the plane
        fp1_pos = plane_line.get_projection(sphere_up.get_center())
        fp2_pos = plane_line.get_projection(sphere_down.get_center())
        
        dot1 = Dot(fp1_pos, color=COLOR_FOCAL, radius=0.08)
        dot2 = Dot(fp2_pos, color=COLOR_FOCAL, radius=0.08)
        
        lbl1 = Text("F1", font_size=16, color=COLOR_FOCAL).next_to(dot1, RIGHT, buff=0.1)
        lbl2 = Text("F2", font_size=16, color=COLOR_FOCAL).next_to(dot2, LEFT, buff=0.1)
        
        self.play(Create(dot1), Create(dot2))
        self.play(Write(lbl1), Write(lbl2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        hist = Text("Germinal Dandelin (1822)", font_size=20, color=YELLOW)
        # Resolved Issue 34: Repositioned to F5 to avoid overlap
        self.place_at_grid(hist, "F5", scale_factor=0.8)
        
        self.play(Write(hist))
        self.play(Indicate(sphere_up), Indicate(sphere_down))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Connect a point on the plane to the focal points to hint at the definition
        p_on_p = plane_line.point_from_proportion(0.4)
        dot_p = Dot(p_on_p, color=WHITE, radius=0.06)
        conn1 = Line(dot1.get_center(), p_on_p, color=YELLOW, stroke_width=2)
        conn2 = Line(dot2.get_center(), p_on_p, color=YELLOW, stroke_width=2)
        
        self.play(FadeIn(dot_p))
        self.play(Create(conn1), Create(conn2))
        self.play(Indicate(conn1), Indicate(conn2))
        
        self.wait(2)
