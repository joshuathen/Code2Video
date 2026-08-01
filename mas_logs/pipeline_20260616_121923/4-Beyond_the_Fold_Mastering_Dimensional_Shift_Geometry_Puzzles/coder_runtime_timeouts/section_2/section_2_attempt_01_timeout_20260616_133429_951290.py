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
        title = "Shadows of Truth: The Logic of Projection"
        lines = [
            "Higher dimensions cast shadows onto lower ones.",
            "A rotating cube creates complex 2D shadows.",
            "Shadows reveal shapes from many different angles."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)

        # Issue 33: Intro horizontal line alignment
        intro_line = Line(LEFT, RIGHT, color=WHITE)
        self.place_at_grid(intro_line, 'C4', scale_factor=0.6)
        self.play(Create(intro_line))
        self.wait(1)
        self.play(FadeOut(intro_line))

        # Issue 27, 31: Cube Asset Integration and Placement
        # Load asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/cube.svg
        cube_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/cube.svg")
        cube_asset.set_color(WHITE)
        self.place_in_area(cube_asset, 'A2', 'C5', scale_factor=1.0)
        
        # Issue 32: Shadow Placement (placed below in E2:F5)
        shadow_poly = Polygon(*[ORIGIN]*4, color="#FFFF00", fill_opacity=0.4, stroke_width=1)
        self.place_in_area(shadow_poly, 'E2', 'F5', scale_factor=0.8)
        shadow_center = shadow_poly.get_center()

        # Rotation and Projection Trackers
        rot_x = ValueTracker(0.4)
        rot_y = ValueTracker(0.4)

        # Cube Geometry
        vertices_3d = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ]) * 0.5

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        def rotate_point(point, ax, ay):
            y_rot = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
            x_rot = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
            return np.dot(x_rot, np.dot(y_rot, point))

        def project_point(point, offset):
            # 2D perspective simulation
            z_factor = 0.3
            x = point[0] + point[2] * z_factor
            y = point[1] + point[2] * z_factor
            return np.array([x, y, 0]) + offset

        cube_lines = VGroup(*[Line(color=WHITE, stroke_width=2) for _ in edges])
        # Position wireframe at same spot as asset for transformation
        self.place_in_area(cube_lines, 'A2', 'C5', scale_factor=1.0)
        cube_center = cube_lines.get_center()

        def update_cube(m):
            ax, ay = rot_x.get_value(), rot_y.get_value()
            for i, (p1, p2) in enumerate(edges):
                v1 = rotate_point(vertices_3d[p1], ax, ay)
                v2 = rotate_point(vertices_3d[p2], ax, ay)
                m[i].set_points_as_corners([project_point(v1, cube_center), project_point(v2, cube_center)])

        def update_shadow(m):
            ax, ay = rot_x.get_value(), rot_y.get_value()
            proj_pts = []
            for v in vertices_3d:
                rotated = rotate_point(v, ax, ay)
                # Shadow projection: squash height and tilt for depth
                shadow_v = np.array([rotated[0], rotated[2] * 0.4, 0]) 
                proj_pts.append(shadow_v + shadow_center)
            # Take one face for representative shadow morphing
            idx_list = [0, 1, 5, 4]
            m.set_points_as_corners([proj_pts[i] for i in idx_list] + [proj_pts[idx_list[0]]])

        cube_lines.add_updater(update_cube)
        shadow_poly.add_updater(update_shadow)

        # Intro visuals
        self.play(FadeIn(cube_asset))
        self.wait(1)
        # Transformation from Asset to rotating wireframe (Issue 27)
        self.play(
            ReplacementTransform(cube_asset, cube_lines), 
            FadeIn(shadow_poly)
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Perform Rotation to show morphing shadow
        self.play(
            rot_x.animate.set_value(PI + 0.4),
            rot_y.animate.set_value(PI + 0.4),
            run_time=4,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Continue rotation to reveal more angles
        self.play(
            rot_x.animate.set_value(2 * PI + PI/4),
            rot_y.animate.set_value(PI/2),
            run_time=4,
            rate_func=smooth
        )
        self.wait(2)
