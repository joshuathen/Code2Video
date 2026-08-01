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
        self.setup_layout("Column Space: The Footprint of the Transformation", [
            "Outputs are restricted to the span of the columns.",
            "This reachable region is called the Column Space.",
            "A 3x2 matrix fills a 2D pane in 3D."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Line 1: Outputs are restricted to the span of the columns.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Define projection for a "3D" look in 2D
        def project_3d(x, y, z):
            # Isometric-like projection
            return np.array([x + 0.4 * z, y + 0.3 * z, 0])

        # Create a 3D wireframe cube vertices
        v = []
        for z in [-1, 1]:
            for y in [-1, 1]:
                for x in [-1, 1]:
                    v.append(project_3d(x, y, z))
        
        # Connections for cube edges
        connections = [
            (0, 1), (1, 3), (3, 2), (2, 0), # back face
            (4, 5), (5, 7), (7, 6), (6, 4), # front face
            (0, 4), (1, 5), (2, 6), (3, 7)  # edges connecting faces
        ]
        
        cube_edges = VGroup(*[Line(v[i], v[j], color=WHITE, stroke_width=2) for i, j in connections])
        
        # Axes guide
        axes = VGroup(
            Line(v[0], project_3d(1, -1, -1), color=GRAY_B, stroke_width=1),
            Line(v[0], project_3d(-1, 1, -1), color=GRAY_B, stroke_width=1),
            Line(v[0], project_3d(-1, -1, 1), color=GRAY_B, stroke_width=1)
        )
        
        # Asset integration: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/tank.svg]
        # Use the tank icon to represent the 3D 'aquarium'
        tank = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tank.svg", color=WHITE, fill_opacity=0.1)
        
        cube_viz = VGroup(tank, cube_edges, axes)
        # Apply positioning fixes from Issue 31 and 32
        # Use B1-F6 and scale 0.7 to avoid title overlap and screen cutoff
        self.place_in_area(cube_viz, "B1", "F6", scale_factor=0.7)
        
        self.play(Create(cube_edges), FadeIn(tank), Create(axes))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: This reachable region is called the Column Space.
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # Re-map points to the cube's actual position on screen
        def get_rel_point(x, y, z):
            p = project_3d(x, y, z)
            # Offset by the group's center shift (excluding tank for center calculation)
            v_center = np.mean(v, axis=0)
            return p - v_center + (cube_viz.get_center() if not hasattr(cube_viz, "submobjects") else cube_edges.get_center())

        origin = get_rel_point(-1, -1, -1)
        vec1_target = get_rel_point(0.8, -0.2, 0.4)
        vec2_target = get_rel_point(-0.2, 0.9, 0.3)
        
        # Basis vectors (#00FF00 and #FF00FF)
        vec1 = Arrow(origin, vec1_target, color="#00FF00", buff=0, tip_length=0.15)
        vec2 = Arrow(origin, vec2_target, color="#FF00FF", buff=0, tip_length=0.15)
        
        self.play(GrowArrow(vec1), GrowArrow(vec2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: A 3x2 matrix fills a 2D pane in 3D.
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        
        # Semi-transparent 2D plane (Column Space)
        d1 = vec1_target - origin
        d2 = vec2_target - origin
        
        plane_pts = [
            origin - 0.3*d1 - 0.3*d2,
            origin + 1.8*d1 - 0.3*d2,
            origin + 1.8*d1 + 1.8*d2,
            origin - 0.3*d1 + 1.8*d2
        ]
        
        col_space = Polygon(*plane_pts, color="#00FFFF", fill_opacity=0.3, stroke_width=1)
        
        # Label for the Column Space
        label_pos = self.grid["C5"] # Near the plane area
        label = Text("Column Space", font_size=18, color="#00FFFF")
        self.place_at_grid(label, "C5", scale_factor=0.8)
        
        self.play(FadeIn(col_space), FadeIn(label))
        self.wait(2)
