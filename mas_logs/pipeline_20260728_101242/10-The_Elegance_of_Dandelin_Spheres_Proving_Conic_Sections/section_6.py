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
        # Initial Setup
        title = "Universal Application: Parabolas and Hyperbolas"
        lines = [
            "Tilting the plane further changes the intersection's shape.",
            "Spheres still reveal the logic of parabolas and hyperbolas.",
            "Geometry and symmetry unite all conic sections."
        ]
        self.setup_layout(title, lines)

        # Colors
        ELLIPSE_COLOR = "#00FF00"
        PARABOLA_COLOR = "#FFA500"
        HYPERBOLA_COLOR = "#FFC0CB"
        CONE_COLOR = BLUE_D
        
        # Assets
        CONE_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg"
        SPHERE_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg"

        # Initialize Mobjects
        cone_upper = SVGMobject(CONE_ASSET).set_color(CONE_COLOR)
        # Double cone for hyperbola later
        cone_lower = SVGMobject(CONE_ASSET).set_color(CONE_COLOR).rotate(PI)
        
        # Position the upper cone (main visual)
        self.place_in_area(cone_upper, 'A1', 'F6', scale_factor=1.5)
        # Shift slightly up to make room for lower cone later
        cone_upper.shift(UP * 0.5)
        cone_lower.next_to(cone_upper, DOWN, buff=0).shift(UP * 0.2) # Connect at vertex

        # Plane line representation
        tilt_line = Line(LEFT * 1.5, RIGHT * 1.5, color=ELLIPSE_COLOR, stroke_width=8)
        self.place_in_area(tilt_line, 'A1', 'F6', scale_factor=1.0)
        tilt_line.shift(UP * 1.2) # Starting position for ellipse intersection

        # Spheres group for easier grid positioning as requested in issue 40
        s1 = SVGMobject(SPHERE_ASSET).set_color(WHITE).scale(0.3)
        s2 = SVGMobject(SPHERE_ASSET).set_color(WHITE).scale(0.6)
        spheres_group = VGroup(s1, s2)
        self.place_in_area(spheres_group, 'B2', 'E5', scale_factor=0.7)
        # Position them relative to the cone and plane
        s1.move_to(cone_upper.get_bottom() + UP * 0.6)
        s2.move_to(cone_upper.get_top() + DOWN * 0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(ELLIPSE_COLOR)
        
        self.play(Create(cone_upper), run_time=1)
        self.play(Create(tilt_line), run_time=1)
        self.play(FadeIn(spheres_group), run_time=1)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Transition to Parabola: Tilt plane parallel to slant line
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(PARABOLA_COLOR)
        
        # Parabola state: one sphere, plane parallel to side
        self.play(
            tilt_line.animate.set_color(PARABOLA_COLOR).rotate(45*DEGREES).shift(DOWN * 0.5 + RIGHT * 0.3),
            s1.animate.scale(1.2).shift(UP * 0.2 + RIGHT * 0.1),
            FadeOut(s2),
            run_time=2.5
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Transition to Hyperbola: Tilt plane further and show double cone
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HYPERBOLA_COLOR)
        
        # Bring in the second cone
        self.play(Create(cone_lower), run_time=1.5)
        
        # Create second sphere for the lower nappe
        s2_new = SVGMobject(SPHERE_ASSET).set_color(WHITE).scale(0.4)
        s2_new.move_to(cone_lower.get_top() + DOWN * 0.6)

        # Rotate plane to vertical and adjust spheres
        self.play(
            tilt_line.animate.set_color(HYPERBOLA_COLOR).set_angle(PI/2).move_to(cone_upper.get_bottom() + RIGHT * 0.5),
            s1.animate.scale(0.8).move_to(cone_upper.get_center() + DOWN * 0.2),
            FadeIn(s2_new),
            run_time=2.5
        )
        self.wait(3)
