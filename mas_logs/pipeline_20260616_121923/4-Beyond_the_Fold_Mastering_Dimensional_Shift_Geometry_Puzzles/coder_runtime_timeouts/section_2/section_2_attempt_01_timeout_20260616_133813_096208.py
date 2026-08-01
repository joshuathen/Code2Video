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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup layout
        title_text = "Shadows of Truth: The Logic of Projection"
        lecture_lines = [
            "Higher dimensions cast shadows onto lower ones.",
            "A rotating cube creates complex 2D shadows.",
            "Shadows reveal shapes from many different angles."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Pre-calculation for Cube Geometry
        vertices_3d = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ]) * 0.4

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        # Tracking variables
        rot_tracker = ValueTracker(0)

        def get_rotated_projection(point, angle):
            # Rotate around Y and X for perspective
            ay = angle
            ax = angle * 0.5
            y_rot = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
            x_rot = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
            p = np.dot(x_rot, np.dot(y_rot, point))
            # Perspective project (simple)
            z_factor = 0.3
            return np.array([p[0] + p[2]*z_factor, p[1] + p[2]*z_factor, 0])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Issue 33: Intro line alignment
        intro_line = Line(LEFT, RIGHT, color=WHITE)
        self.place_at_grid(intro_line, 'C4', scale_factor=0.6)
        self.play(Create(intro_line), run_time=1)
        self.play(FadeOut(intro_line), run_time=0.5)

        # Issue 27: Asset Integration
        cube_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/cube.svg")
        cube_asset.set_color(WHITE)
        # Issue 31: Align to A2:C5
        self.place_in_area(cube_asset, 'A2', 'C5', scale_factor=1.2)
        cube_center = cube_asset.get_center()

        self.play(DrawBorderThenFill(cube_asset))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Build wireframe cube (will replace SVG)
        cube_wireframe = VGroup(*[Line(color=WHITE, stroke_width=2) for _ in edges])
        cube_wireframe.move_to(cube_center)

        # Shadow Setup (Issue 32: Align to E2:F5)
        shadow_poly = Polygon(*[ORIGIN]*4, color=YELLOW, fill_opacity=0.3, stroke_width=1)
        self.place_in_area(shadow_poly, 'E2', 'F5', scale_factor=1.0)
        shadow_center = shadow_poly.get_center()

        def update_cube(m):
            angle = rot_tracker.get_value()
            for i, (p1, p2) in enumerate(edges):
                v1 = get_rotated_projection(vertices_3d[p1], angle)
                v2 = get_rotated_projection(vertices_3d[p2], angle)
                m[i].set_points_as_corners([v1 + cube_center, v2 + cube_center])

        def update_shadow(m):
            angle = rot_tracker.get_value()
            # To simulate a morphing shadow, project some vertices to the "floor"
            # We'll use 4 representative vertices to form a parallelogram as requested
            indices = [0, 1, 5, 4]
            pts = []
            for idx in indices:
                v = vertices_3d[idx]
                ay, ax = angle, angle * 0.5
                y_rot = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
                x_rot = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
                rotated = np.dot(x_rot, np.dot(y_rot, v))
                # Flatten Y for shadow effect
                pts.append(np.array([rotated[0], rotated[2]*0.5, 0]) + shadow_center)
            m.set_points_as_corners([*pts, pts[0]])

        cube_wireframe.add_updater(update_cube)
        shadow_poly.add_updater(update_shadow)

        self.play(
            ReplacementTransform(cube_asset, cube_wireframe),
            FadeIn(shadow_poly)
        )
        
        # Start rotation
        self.play(rot_tracker.animate.set_value(PI), run_time=4, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Continue rotation with smooth deceleration
        self.play(rot_tracker.animate.set_value(2 * PI + PI/4), run_time=4, rate_func=smooth)
        self.wait(2)
