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
        # Setup layout with title and lecture lines
        self.setup_layout("Prerequisite: The Ice Cream Cone Lemma", [
            "- Tangents from a point to a sphere are equal.",
            "- This is known as the \"Ice Cream Cone Lemma.\"",
            "- Distances from the vertex to the contact circle match."
        ])
        
        # Initial lecture colors - start dimmed for progressive highlight
        self.lecture.set_color(GRAY)
        
        # Visual Element Colors
        sphere_color = "#FFFFFF"
        point_p_color = "#FF0000"
        tangent_color = "#FFFFFF"
        mark_color = "#FFFFFF"
        
        # Assets
        icecream_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/icecream.svg"
        cone_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg"
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Create a circle for geometry calculations but use the icecream SVG visually
        sphere_geom = Circle(radius=1.2, color=sphere_color)
        self.place_in_area(sphere_geom, 'C4', 'E6') # Fixed positioning (Issue 35)
        
        icecream_svg = SVGMobject(icecream_asset_path).set_color(sphere_color)
        self.place_in_area(icecream_svg, 'C4', 'E6', scale_factor=1.6)
        
        # Create point P outside the sphere
        p_dot = Dot(color=point_p_color)
        self.place_at_grid(p_dot, 'D2') # Fixed positioning (Issue 34)
        p_label = MathTex("P", color=point_p_color).next_to(p_dot, LEFT, buff=0.1)
        
        self.play(FadeIn(icecream_svg), FadeIn(p_dot), Write(p_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(RED))
        
        # Geometry setup for tangents
        c = sphere_geom.get_center()
        p = p_dot.get_center()
        r = sphere_geom.radius
        cp_vec = p - c
        dist_cp = np.linalg.norm(cp_vec)
        
        # Calculate tangent points on the circle
        angle_cp = np.arctan2(cp_vec[1], cp_vec[0])
        angle_ct = np.arccos(r / dist_cp)
        
        t1_pos = c + r * np.array([np.cos(angle_cp + angle_ct), np.sin(angle_cp + angle_ct), 0])
        t2_pos = c + r * np.array([np.cos(angle_cp - angle_ct), np.sin(angle_cp - angle_ct), 0])
        
        l1 = Line(p, t1_pos, color=tangent_color)
        l2 = Line(p, t2_pos, color=tangent_color)
        
        # Load and place cone asset
        cone_svg = SVGMobject(cone_asset_path).set_color(tangent_color)
        self.place_in_area(cone_svg, 'C2', 'E4', scale_factor=1.0)
        
        self.play(Create(l1), Create(l2), FadeIn(cone_svg))
        self.play(Indicate(p_dot, color=RED), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # Flash both tangent segments to emphasize their equal lengths
        self.play(
            Flash(l1, color=WHITE, line_length=0.3, flash_radius=0.5),
            Flash(l2, color=WHITE, line_length=0.3, flash_radius=0.5)
        )
        
        # Equality markers (ticks)
        def create_tick(line):
            mid = line.get_center()
            vec = line.get_unit_vector()
            # Calculate a normal vector for the tick
            normal = np.array([-vec[1], vec[0], 0])
            return Line(mid - 0.12 * normal, mid + 0.12 * normal, color=mark_color, stroke_width=3)

        tick1 = create_tick(l1)
        tick2 = create_tick(l2)
        
        self.play(Create(tick1), Create(tick2))
        self.wait(2)
