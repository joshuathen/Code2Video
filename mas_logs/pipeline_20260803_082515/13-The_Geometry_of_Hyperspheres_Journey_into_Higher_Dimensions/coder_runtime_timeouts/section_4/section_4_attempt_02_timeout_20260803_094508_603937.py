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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "The 'Spiky' Box: The Touching Spheres Problem",
            [
                "Four circles in a square leave a central gap.",
                "More dimensions squeeze the inner sphere even further.",
                "In ten dimensions, the inner sphere pokes outside.",
                "It stretches through gaps in the outer spheres.",
                "High-dimensional geometry defies our 3D intuition completely."
            ]
        )

        sphere_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg"
        box_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg"

        # === Animation for Lecture Line 1 ===
        # 2D: 4 white circles (#FFFFFF) in corners of a square; a tiny green circle (#00FF00) in the middle.
        self.lecture[0].set_color(YELLOW)
        
        # We use a base scale where box radius is 2, sphere radius is 1.
        # This makes calculation of inner radius r = sqrt(n) - 1 easy.
        
        box_2d = Square(side_length=4.0, color=WHITE)
        self.place_in_area(box_2d, "B2", "E5", scale_factor=0.7)
        
        corner_offsets = [
            np.array([-1, 1, 0]),
            np.array([1, 1, 0]),
            np.array([-1, -1, 0]),
            np.array([1, -1, 0])
        ]
        
        # Note: box_2d was scaled by 0.7 during place_in_area. 
        # We should calculate offsets based on the scaled box.
        scale_val = box_2d.width / 4.0
        
        white_circles = VGroup(*[
            Circle(radius=1.0 * scale_val, color=WHITE, stroke_width=2).move_to(box_2d.get_center() + offset * scale_val)
            for offset in corner_offsets
        ])
        
        inner_radius_2d = (np.sqrt(2) - 1) * scale_val
        inner_circle = Circle(radius=inner_radius_2d, color="#00FF00", fill_opacity=0.6, stroke_width=0)
        inner_circle.move_to(box_2d.get_center())
        
        self.play(Create(box_2d), Create(white_circles))
        self.play(FadeIn(inner_circle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # 3D: 8 white spheres [Asset: sphere.svg] in corners of a cube; a larger green sphere in the middle.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Transition to 3D representation using sphere assets
        # To avoid recreate/scale issues, we'll build the 3D group first.
        
        def get_sphere_svg(color=WHITE, scale=1.0):
            # sphere.svg likely has some internal structure, set_color might work on parts
            s = SVGMobject(sphere_path).scale(scale)
            # Find any parts that look like they should be colored
            for part in s.submobjects:
                part.set_color(color)
            return s

        # For 3D, we'll show a perspective box
        box_3d_front = Square(side_length=4.0, color=WHITE, stroke_opacity=0.4)
        box_3d_back = Square(side_length=4.0, color=WHITE, stroke_opacity=0.4).shift(0.5*UP + 0.5*RIGHT)
        box_3d_group = VGroup(box_3d_front, box_3d_back)
        for c in [UL, UR, DL, DR]:
            box_3d_group.add(Line(box_3d_front.get_corner(c), box_3d_back.get_corner(c), color=WHITE, stroke_opacity=0.4))
        
        self.place_in_area(box_3d_group, "B2", "E5", scale_factor=0.6)
        
        # 8 corner spheres
        spheres_3d = VGroup()
        scale_3d = box_3d_front.width / 4.0
        for b in [box_3d_front, box_3d_back]:
            for offset in corner_offsets:
                spheres_3d.add(get_sphere_svg(WHITE, scale=scale_3d * 0.5).move_to(b.get_center() + offset * scale_3d))

        inner_radius_3d = (np.sqrt(3) - 1) * scale_3d
        # Inner sphere SVG
        inner_sphere_svg = get_sphere_svg("#00FF00", scale=inner_radius_3d * 0.5)
        inner_sphere_svg.move_to(box_3d_group.get_center())

        self.play(
            ReplacementTransform(box_2d, box_3d_group),
            FadeOut(white_circles),
            FadeOut(inner_circle),
            FadeIn(spheres_3d),
            FadeIn(inner_sphere_svg)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display formula r_inner = sqrt(n) - 1 in #00FFFF.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        formula = MathTex(r"r_{\text{inner}} = \sqrt{n} - 1", color="#00FFFF")
        # Issue 27: self.place_in_area(formula, 'A2', 'A3', scale_factor=0.8)
        self.place_in_area(formula, "A2", "A3", scale_factor=0.8)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Animate the green sphere growing as 'n' increases on a slider.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # We'll return to a clearer 2D cross-section for the growth to show poking out.
        # But we'll use the box asset if provided.
        box_asset = SVGMobject(box_path).set_color(WHITE)
        self.place_in_area(box_asset, "B2", "E5", scale_factor=0.7)
        
        # Re-calc scale for SVG box. Assuming SVG box corresponds to the [-2, 2] square.
        # If it's a cube asset, we'll treat its projection.
        box_scale_val = box_asset.width / 4.0
        
        # 4 corner spheres (using SVG)
        corner_spheres_svg = VGroup(*[
            get_sphere_svg(WHITE, scale=box_scale_val * 0.5).move_to(box_asset.get_center() + offset * box_scale_val)
            for offset in corner_offsets
        ])
        
        # Growing sphere
        growing_sphere = get_sphere_svg("#00FF00", scale=(np.sqrt(3)-1) * box_scale_val * 0.5)
        growing_sphere.move_to(box_asset.get_center())
        
        n_tracker = ValueTracker(3)
        n_label = MathTex("n = ", color=WHITE)
        n_value = DecimalNumber(3, num_decimal_places=0, color=WHITE)
        n_group = VGroup(n_label, n_value).arrange(RIGHT)
        # Issue 28: self.place_in_area(n_group, 'A4', 'A5', scale_factor=0.8)
        self.place_in_area(n_group, "A4", "A5", scale_factor=0.8)

        # SVG doesn't have a radius, we use scale. 
        # Initial scale was for n=3: (sqrt(3)-1) * box_scale_val * 0.5
        # To get scale for n, we scale by (sqrt(n)-1)/(sqrt(3)-1)
        base_r = np.sqrt(3) - 1
        
        def update_sphere(mob):
            n = n_tracker.get_value()
            r = np.sqrt(n) - 1
            mob.become(get_sphere_svg("#00FF00" if r <= 2 else "#FF0000", scale=r * box_scale_val * 0.5))
            mob.move_to(box_asset.get_center())

        self.play(
            FadeOut(box_3d_group),
            FadeOut(spheres_3d),
            FadeOut(inner_sphere_svg),
            FadeIn(box_asset),
            FadeIn(corner_spheres_svg),
            FadeIn(growing_sphere),
            FadeIn(n_group)
        )
        
        growing_sphere.add_updater(update_sphere)
        n_value.add_updater(lambda m: m.set_value(n_tracker.get_value()))
        
        self.play(n_tracker.animate.set_value(9), run_time=3, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # At n=10, the green sphere's edges extend beyond the white box edges (#FF0000).
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        self.play(n_tracker.animate.set_value(10), run_time=2)
        
        # Highlight the overlap
        overlap_text = Text("Pokes outside!", font_size=20, color="#FF0000")
        # Issue 29: self.place_in_area(overlap_text, 'F3', 'F4', scale_factor=0.8)
        self.place_in_area(overlap_text, "F3", "F4", scale_factor=0.8)
        self.play(Write(overlap_text))
        
        self.wait(2)
        
        growing_sphere.remove_updater(update_sphere)
        n_value.remove_updater(lambda m: m.set_value(n_tracker.get_value()))
