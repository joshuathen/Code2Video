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

class Section6Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "This logic applies to more than just ellipses.",
            "Tilt the plane to see a parabola's single focus.",
            "Steeper tilts create the two branches of a hyperbola.",
            "One sphere moves above the vertex for the hyperbola.",
            "Dandelin spheres unify all conics through one geometric proof."
        ]
        self.setup_layout("Universal Application", lecture_lines)

        # Assets
        sphere_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg"

        # Cone lines (forming two nappes in cross-section)
        # Generators of the cone
        # Intersection is roughly (3.0, -0.3)
        l1 = Line(self.grid["A1"], self.grid["F6"], color=GRAY)
        l2 = Line(self.grid["A6"], self.grid["F1"], color=GRAY)
        cone = VGroup(l1, l2)

        # === Animation for Lecture Line 1 ===
        # "This logic applies to more than just ellipses."
        self.lecture[0].set_color(YELLOW)
        self.play(Create(cone))
        
        # Initial ellipse-like plane
        plane = Line(self.grid["C1"], self.grid["E6"], color="#00FFFF")
        self.play(Create(plane))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Tilt the plane to see a parabola's single focus."
        self.play(
            self.lecture[0].animate.set_color(WHITE), 
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Parabola plane: parallel to generator l2 (slope 1)
        parabola_plane_target = Line(self.grid["E2"], self.grid["A6"], color="#00FFFF")
        
        # Dandelin Sphere for parabola (using SVG asset)
        # Position fixed to E5 per Issue 41
        sphere_para = SVGMobject(sphere_asset).set_color(YELLOW)
        self.place_at_grid(sphere_para, "E5", scale_factor=0.6)
        
        # Focal point (tangency point)
        focus_para = Dot(color=PINK).move_to(sphere_para.get_center() + 0.3*LEFT + 0.3*UP)
        
        self.play(Transform(plane, parabola_plane_target))
        self.play(FadeIn(sphere_para), FadeIn(focus_para))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Steeper tilts create the two branches of a hyperbola."
        self.play(
            self.lecture[1].animate.set_color(WHITE), 
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Hyperbola plane: Vertical
        hyperbola_plane_target = Line(self.grid["A4"], self.grid["F4"], color="#00FFFF")
        self.play(Transform(plane, hyperbola_plane_target))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "One sphere moves above the vertex for the hyperbola."
        self.play(
            self.lecture[2].animate.set_color(WHITE), 
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        # Two spheres for hyperbola (using SVG asset)
        # Positions fixed to E5 and B5 per Issues 42 and 43
        sphere_hyp_bottom = SVGMobject(sphere_asset).set_color(YELLOW)
        self.place_at_grid(sphere_hyp_bottom, "E5", scale_factor=0.6)
        
        sphere_hyp_top = SVGMobject(sphere_asset).set_color(YELLOW)
        self.place_at_grid(sphere_hyp_top, "B5", scale_factor=0.6)
        
        # Foci for hyperbola branches
        focus_hyp_bottom = Dot(color=PINK).move_to(sphere_hyp_bottom.get_center() + 0.4*LEFT)
        focus_hyp_top = Dot(color=PINK).move_to(sphere_hyp_top.get_center() + 0.4*LEFT)
        
        self.play(
            ReplacementTransform(sphere_para, sphere_hyp_bottom),
            ReplacementTransform(focus_para, focus_hyp_bottom),
            FadeIn(sphere_hyp_top),
            FadeIn(focus_hyp_top)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Dandelin spheres unify all conics through one geometric proof."
        self.play(
            self.lecture[3].animate.set_color(WHITE), 
            self.lecture[4].animate.set_color(YELLOW)
        )
        self.wait(1)
        
        # Indicate the unified configuration
        self.play(
            Indicate(plane, color="#00FFFF"), 
            Indicate(sphere_hyp_bottom, color=YELLOW), 
            Indicate(sphere_hyp_top, color=YELLOW)
        )
        self.wait(2)
        
        # Final cleanup
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
