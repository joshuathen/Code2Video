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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup the layout with mandatory lecture lines
        lecture_lines_text = [
            "Imagine any simple, closed loop in a plane.",
            "Place four points randomly along this wobbly boundary.",
            "These points can form various four-sided shapes.",
            "But can we always find a perfect square?",
            "This unsolved mystery is the Toeplitz Conjecture."
        ]
        self.setup_layout("The Hook: The Pirate's Square Treasure", lecture_lines_text)

        # Colors
        COLOR_CURVE = "#ADD8E6"
        COLOR_QUAD = "#FFFF00"
        COLOR_SQUARE = "#00FF00"
        COLOR_TEXT = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_CURVE))
        
        # Create an irregular Jordan curve that contains a potential square
        sq_verts = [
            np.array([1.2, 1.2, 0]), 
            np.array([-1.2, 1.2, 0]), 
            np.array([-1.2, -1.2, 0]), 
            np.array([1.2, -1.2, 0])
        ]
        
        curve_pts = []
        for i in range(4):
            curve_pts.append(sq_verts[i])
            mid = (sq_verts[i] + sq_verts[(i+1)%4]) / 2
            offset = np.array([np.sin(i*2)*0.4, np.cos(i*3)*0.4, 0])
            curve_pts.append(mid + offset)
            
        curve = VMobject()
        curve.set_points_as_corners(curve_pts + [curve_pts[0]])
        curve.make_smooth()
        curve.set_color(COLOR_CURVE)
        
        # Issue 39: Fixing scale factor from 1.2 to 1.0 to avoid tight margins
        self.place_in_area(curve, "B2", "E5", scale_factor=1.0)
        
        self.play(Create(curve))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_QUAD))
        
        # Initial "random" positions for the dots
        start_props = [0.1, 0.4, 0.6, 0.9]
        dots = VGroup(*[Dot(curve.point_from_proportion(p), color=COLOR_QUAD) for p in start_props])
        
        self.play(FadeIn(dots))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_QUAD))
        
        # Create quadrilateral connecting the dots
        quad = VMobject(color=COLOR_QUAD, stroke_width=2)
        def get_quad_points():
            return [d.get_center() for d in dots]
        quad.add_updater(lambda m: m.set_points_as_corners(get_quad_points() + [get_quad_points()[0]]))
        
        self.play(Create(quad))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_SQUARE))

        # Target proportions for a square vertices on the curve
        sq_props = [0.0, 0.25, 0.5, 0.75]
        
        animations = []
        for i in range(4):
            start_p = start_props[i]
            end_p = sq_props[i]
            
            def get_subpath(s, e):
                pts = []
                steps = 10
                for step in range(steps + 1):
                    alpha = step / steps
                    prop = s + (e - s) * alpha
                    pts.append(curve.point_from_proportion(prop % 1.0))
                return VMobject().set_points_as_corners(pts)

            subpath = get_subpath(start_p, end_p)
            animations.append(MoveAlongPath(dots[i], subpath))

        self.play(*animations, run_time=2)
        
        # Transform to perfect green square
        quad.clear_updaters()
        final_square = Polygon(*[d.get_center() for d in dots], color=COLOR_SQUARE, stroke_width=4)
        
        self.play(
            ReplacementTransform(quad, final_square),
            dots.animate.set_color(COLOR_SQUARE)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_TEXT))
        
        conjecture_text = Text("Toeplitz Conjecture", font_size=24, color=COLOR_TEXT)
        # Issue 38: Place in area F3-F5 with scale 0.8
        self.place_in_area(conjecture_text, "F3", "F5", scale_factor=0.8)
        
        self.play(FadeIn(conjecture_text))
        self.wait(2)
