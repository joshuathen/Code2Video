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
        # Data from storyboard
        title = "The Shadow Puzzle: 3D to 2D Projection"
        lecture_lines = [
            "Higher dimensions lose information when projected onto lower ones.",
            "A cube's 2D shadow changes shape as it rotates.",
            "We can identify objects by studying their shifting footprints."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # Colors
        CUBE_COLOR = "#FFFFFF"
        SHADOW_COLOR = "#808080"
        HIGHLIGHT_COLOR = "#FFFF00"
        
        # Helper for Convex Hull (Monotone Chain algorithm)
        def get_convex_hull(points):
            n = len(points)
            if n <= 2: return points
            # Sort by x then y
            pts = sorted(points, key=lambda p: (p[0], p[1]))
            
            def cross_product(o, a, b):
                return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            
            upper = []
            for p in pts:
                while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
                    upper.pop()
                upper.append(p)
            
            lower = []
            for p in reversed(pts):
                while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
                    lower.pop()
                lower.append(p)
            
            return upper[:-1] + lower[:-1]

        # === Animation for Lecture Line 1 ===
        # Intro: Higher dimensions lose information when projected onto lower ones.
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # 3D Cube Vertices (wireframe)
        v_scale = 0.6
        raw_vertices = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ]) * v_scale
        
        edges = [
            (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
        ]
        
        # Container for cube lines
        cube_group = VGroup(*[Line(ORIGIN, RIGHT, color=CUBE_COLOR, stroke_width=2) for _ in edges])
        
        # Trackers for rotation
        theta = ValueTracker(0) # Y-axis
        phi = ValueTracker(0)   # X-axis
        
        def update_cube(mob):
            t = theta.get_value()
            p = phi.get_value()
            # Rotation matrices
            ry = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
            rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
            mat = np.dot(rx, ry)
            
            current_v = [np.dot(mat, v) for v in raw_vertices]
            for i, (u, v) in enumerate(edges):
                mob[i].set_points_as_corners([current_v[u] + self.grid["C2"], current_v[v] + self.grid["C2"]])

        cube_group.add_updater(update_cube)
        
        # Shadow Mobject
        shadow_poly = Polygon(*[ORIGIN]*6, color=SHADOW_COLOR, fill_opacity=0.6, stroke_width=1)
        
        def update_shadow(mob):
            t = theta.get_value()
            p = phi.get_value()
            ry = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
            rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
            mat = np.dot(rx, ry)
            
            # Project to XY plane (take first two coords)
            proj = [np.dot(mat, v)[:2] for v in raw_vertices]
            hull = get_convex_hull(proj)
            pts3d = [np.array([pt[0], pt[1], 0]) + self.grid["C5"] for pt in hull]
            mob.set_points_as_corners([*pts3d, pts3d[0]])

        shadow_poly.add_updater(update_shadow)
        
        # Labels - Addressing Issues 26, 27, 28
        cube_label = Text("3D Object", font_size=20, color=CUBE_COLOR)
        shadow_label = Text("2D Projection", font_size=20, color=SHADOW_COLOR)
        self.place_in_area(cube_label, "A1", "A3", scale_factor=0.8)
        self.place_in_area(shadow_label, "A4", "A6", scale_factor=0.8)
        
        self.add(cube_group, shadow_poly, cube_label, shadow_label)
        self.play(
            FadeIn(cube_group), 
            FadeIn(shadow_poly), 
            Write(cube_label), 
            Write(shadow_label),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Movement: A cube's 2D shadow changes shape as it rotates.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Rotate to show square to hexagon
        # A hexagon appears near t=PI/4, p=0.615
        self.play(
            theta.animate.set_value(PI/4),
            phi.animate.set_value(0.615),
            run_time=4,
            rate_func=slow_into
        )
        self.wait(1)
        
        # Further rotation to show variety
        self.play(
            theta.animate.increment_value(PI),
            phi.animate.increment_value(PI/2),
            run_time=5,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Conclusion: We can identify objects by studying their shifting footprints.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Highlight the shadow
        self.play(
            shadow_poly.animate.set_stroke(HIGHLIGHT_COLOR, width=3),
            Indicate(shadow_poly, scale_factor=1.1, color=HIGHLIGHT_COLOR),
            run_time=2
        )
        
        # Final rotation loop
        self.play(
            theta.animate.increment_value(PI/2),
            run_time=3,
            rate_func=linear
        )
        self.wait(2)
