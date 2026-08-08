import numpy as np
from manim import *

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
        # Setup layout with title and lecture lines
        self.setup_layout("Laplace's Equation: The State of Equilibrium", [
            "Laplace’s Equation represents a system in steady-state equilibrium.",
            "There is no change over time; values are balanced.",
            "A soap film finds the most relaxed, averaged surface."
        ])
        
        # Helper function to create a "3D" mesh grid in 2D space
        def get_mesh(z_func, tl_pos, br_pos, rows=9, cols=9):
            mesh = VGroup()
            width = br_pos[0] - tl_pos[0]
            height = tl_pos[1] - br_pos[1]
            
            def project(r, c):
                # Normalized coordinates in [0, 1]
                u = c / (cols - 1)
                v = r / (rows - 1)
                x_base = tl_pos[0] + u * width
                y_base = tl_pos[1] - v * height
                z = z_func(u, v)
                # Apply simple isometric projection for 3D look
                return np.array([x_base - z * 0.3, y_base + z * 0.3, 0])

            # Generate grid lines
            for r in range(rows):
                line = VMobject().set_points_as_corners([project(r, c) for c in range(cols)])
                mesh.add(line)
            for c in range(cols):
                line = VMobject().set_points_as_corners([project(r, c) for r in range(rows)])
                mesh.add(line)
            return mesh

        # Define mesh parameters
        tl_mesh = self.grid["B3"]
        br_mesh = self.grid["E6"]
        
        # === Animation for Lecture Line 1 ===
        # Show Laplace's Equation ∇²u = 0 in #FFFFFF.
        laplace_eq = MathTex(r"\nabla^2 u = 0", color="#FFFFFF")
        # Fixed: issue #41 - Repositioned to A4-A6, scale 1.0
        self.place_in_area(laplace_eq, "A4", "A6", scale_factor=1.0)
        
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Write(laplace_eq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate a 3D soap film surface reaching a smooth, 'relaxed' equilibrium shape.
        
        # Asset integration: issue #28
        soap_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/soap.svg")
        self.place_at_grid(soap_icon, "B2", scale_factor=0.6)
        
        # Initial jagged/distorted state
        def z_initial(u, v):
            if 0 < u < 1 and 0 < v < 1:
                return 0.4 * np.sin(u * 12) * np.cos(v * 12)
            return 0
        
        # Relaxed state (a smooth saddle surface which is a solution to Laplace's eq)
        def z_final(u, v):
            return 1.2 * (u - 0.5) * (v - 0.5)

        mesh_distorted = get_mesh(z_initial, tl_mesh, br_mesh)
        mesh_distorted.set_stroke(color="#44AAFF", width=1.5, opacity=0.7)
        
        mesh_relaxed = get_mesh(z_final, tl_mesh, br_mesh)
        mesh_relaxed.set_stroke(color="#44AAFF", width=2, opacity=0.9)
        
        self.play(self.lecture[1].animate.set_color("#44AAFF"))
        self.play(FadeIn(soap_icon))
        self.play(Create(mesh_distorted))
        self.wait(0.5)
        self.play(Transform(mesh_distorted, mesh_relaxed, run_time=2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight a point on the surface showing it is the average of its neighbors.
        
        # Define projection for a specific point and its neighbors in the relaxed state
        def project_final(u, v):
            z = z_final(u, v)
            width = br_mesh[0] - tl_mesh[0]
            height = tl_mesh[1] - br_mesh[1]
            x_base = tl_mesh[0] + u * width
            y_base = tl_mesh[1] - v * height
            return np.array([x_base - z * 0.3, y_base + z * 0.3, 0])

        # Coordinates for the highlighted point (center) and neighbors
        center_uv = (0.5, 0.5)
        step = 1.0 / 8.0 
        
        p_center = project_final(*center_uv)
        p_neighbors = [
            project_final(center_uv[0] + step, center_uv[1]),
            project_final(center_uv[0] - step, center_uv[1]),
            project_final(center_uv[0], center_uv[1] + step),
            project_final(center_uv[0], center_uv[1] - step)
        ]
        
        center_dot = Dot(p_center, color="#FFFF00", radius=0.08)
        neighbor_dots = VGroup(*[Dot(p, color="#00FF00", radius=0.05) for p in p_neighbors])
        arrows = VGroup(*[
            Line(p, p_center, color="#00FF00", stroke_width=2).add_tip(tip_length=0.1) 
            for p in p_neighbors
        ])
        
        avg_label = MathTex(r"u_{i,j} = \frac{1}{4} \sum u_{\text{neighbors}}", color="#FFFF00", font_size=24)
        # Fixed: issue #42 - Repositioned to F4-F6, scale 0.8
        self.place_in_area(avg_label, "F4", "F6", scale_factor=0.8)

        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.play(FadeIn(center_dot))
        self.play(
            AnimationGroup(
                Create(neighbor_dots),
                Create(arrows),
                lag_ratio=0.3
            )
        )
        self.play(Write(avg_label))
        self.wait(3)
