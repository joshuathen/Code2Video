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
        # Initialization
        lines = [
            "This geometric magic works for all conic sections.",
            "Tilt the plane to create parabolas or hyperbolas.",
            "Dandelin spheres always reveal the hidden foci."
        ]
        self.setup_layout("Universal Geometry: Parabolas and Hyperbolas", lines)
        
        # Colors
        COLOR_CONE = "#999999"
        COLOR_PLANE = "#FFFFFF"
        COLOR_CURVE = "#FFFF00"
        COLOR_SPHERE = "#00FF00"
        COLOR_FOCUS = "#FF5555"
        COLOR_HL = "#00FFFF" # Highlighting color for text

        # Assets
        plane_asset_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/plane.svg"
        sphere_asset_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/sphere.svg"

        # Anchor for geometry
        apex = self.grid["D3"]

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_HL))
        
        # Cross-section of a double cone
        line_a = Line(apex + 2.5*UL, apex + 2.5*DR, color=COLOR_CONE)
        line_b = Line(apex + 2.5*UR, apex + 2.5*DL, color=COLOR_CONE)
        double_cone = VGroup(line_a, line_b)
        
        # Initial Plane (Asset) - start in an angled orientation
        plane_svg = SVGMobject(plane_asset_path, color=COLOR_PLANE).scale(0.7)
        plane_svg.move_to(apex + 0.2*UP)
        plane_svg.rotate(20 * DEGREES) # Current angle: 20
        
        self.play(Create(double_cone), FadeIn(plane_svg))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_HL)
        )
        
        # Parabola Position (parallel to side slope -1, angle -45 deg)
        # Fix Issue 41: Place Parabola label at B4
        shape_label = Text("Parabola", font_size=24, color=COLOR_CURVE)
        self.place_at_grid(shape_label, "B4", scale_factor=0.8)
        
        self.play(
            plane_svg.animate.rotate(-65 * DEGREES).move_to(apex + 0.5*RIGHT + 0.5*DOWN),
            FadeIn(shape_label)
        )
        
        # Briefly show Dandelin sphere for Parabola (Issue 28)
        s_para = SVGMobject(sphere_asset_path, color=COLOR_SPHERE).scale(0.3)
        s_para.move_to(apex + 0.5*UP)
        f_para = Dot(apex + 0.5*RIGHT + 0.5*UP, color=COLOR_FOCUS)
        self.play(FadeIn(s_para, f_para))
        self.wait(0.5)
        self.play(FadeOut(s_para, f_para))
        
        # Hyperbola Position (vertical, angle 90 deg)
        # Fix Issue 42: Place Hyperbola label at B6
        hyper_label = Text("Hyperbola", font_size=24, color=COLOR_CURVE)
        self.place_at_grid(hyper_label, "B6", scale_factor=0.8)
        
        self.play(
            plane_svg.animate.rotate(135 * DEGREES).move_to(apex + 1.0*RIGHT),
            Transform(shape_label, hyper_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_HL)
        )
        
        # Dandelin sphere assets for Hyperbola (Issue 28)
        sphere_y_offset = 1.41
        s_top = SVGMobject(sphere_asset_path, color=COLOR_SPHERE).scale(0.4)
        s_top.move_to(apex + UP * sphere_y_offset)
        
        s_bottom = SVGMobject(sphere_asset_path, color=COLOR_SPHERE).scale(0.4)
        s_bottom.move_to(apex + DOWN * sphere_y_offset)
        
        # Foci
        focus_1 = Dot(apex + RIGHT * 1.0 + UP * sphere_y_offset, color=COLOR_FOCUS)
        focus_2 = Dot(apex + RIGHT * 1.0 + DOWN * sphere_y_offset, color=COLOR_FOCUS)
        
        # Fix Issue 40: Place Foci label at D5
        foci_tag = Text("Foci", font_size=22, color=COLOR_FOCUS)
        self.place_at_grid(foci_tag, "D5", scale_factor=0.8)
        
        self.play(
            FadeIn(s_top, s_bottom),
            FadeIn(focus_1, focus_2),
            FadeIn(foci_tag)
        )
        self.wait(2)
        
        # Final cleanup
        self.play(
            FadeOut(double_cone, plane_svg, s_top, s_bottom, focus_1, focus_2, foci_tag, shape_label),
            self.lecture[2].animate.set_color(WHITE)
        )
        self.wait(1)
