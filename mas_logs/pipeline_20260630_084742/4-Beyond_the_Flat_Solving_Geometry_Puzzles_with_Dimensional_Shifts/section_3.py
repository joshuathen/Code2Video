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
        # Initial layout setup
        self.setup_layout(
            "The Shortest Path Puzzle (3D to 2D)", 
            [
                'A spider needs the shortest path across a cube.', 
                'Measuring across 3D surfaces can be quite confusing.', 
                'Unfolding the cube into a 2D net simplifies everything.', 
                'A straight line on the flat net is the answer.', 
                'The 2D shift reveals the most efficient route.'
            ]
        )

        # Projection function for "3D" cube view (isometric-ish)
        def project_3d(point):
            x, y, z = point
            # Small shift for depth
            return np.array([x + 0.4 * z, y + 0.3 * z, 0])

        # Define faces of the unit cube
        # Each face is defined by 4 vertices in 3D
        face_data = [
            {"name": "Bottom", "verts": [[0,0,0], [1,0,0], [1,0,1], [0,0,1]]},
            {"name": "Front",  "verts": [[0,0,1], [1,0,1], [1,1,1], [0,1,1]]},
            {"name": "Top",    "verts": [[0,1,1], [1,1,1], [1,1,0], [0,1,0]]},
            {"name": "Back",   "verts": [[0,1,0], [1,1,0], [1,0,0], [0,0,0]]},
            {"name": "Left",   "verts": [[0,0,0], [0,0,1], [0,1,1], [0,1,0]]},
            {"name": "Right",  "verts": [[1,0,1], [1,0,0], [1,1,0], [1,1,1]]},
        ]

        # Create Mobjects for faces
        faces = VGroup()
        for face in face_data:
            poly = Polygon(*[project_3d(v) for v in face["verts"]], 
                           color=WHITE, stroke_width=2, fill_opacity=0.1)
            poly.face_name = face["name"]
            faces.add(poly)

        # Define dots at original projected positions
        spider_dot = Dot(project_3d([0, 0, 0]), color="#FF0000", radius=0.08)
        fly_dot = Dot(project_3d([1, 1, 1]), color="#0000FF", radius=0.08)
        
        # Group them to place together in the grid area (Issue 39 Fix)
        cube_group = VGroup(faces, spider_dot, fly_dot)
        self.place_in_area(cube_group, "B2", "F6", scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(faces), FadeIn(spider_dot), FadeIn(fly_dot))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Define paths using vertices from the already placed and scaled faces to ensure alignment
        p1 = faces[0].get_vertices()[0] # Bottom (0,0,0)
        p2 = faces[0].get_vertices()[1] # Bottom (1,0,0)
        p3 = faces[2].get_vertices()[2] # Top (1,1,0)
        p4 = faces[2].get_vertices()[1] # Top (1,1,1)
        
        standard_path = VGroup(
            DashedLine(p1, p2, color="#FFFF00"),
            DashedLine(p2, p3, color="#FFFF00"),
            DashedLine(p3, p4, color="#FFFF00")
        )
        
        self.play(Create(standard_path))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        self.play(FadeOut(standard_path))

        # Net positions (normalized to unit square grid)
        net_offsets = {
            "Bottom": [0, -1],
            "Front":  [0, 0],
            "Top":    [0, 1],
            "Back":   [0, 2],
            "Left":   [-1, 0],
            "Right":  [1, 0],
        }

        # Target vertices for the net
        target_faces = VGroup()
        for face_mobject in faces:
            name = face_mobject.face_name
            offset = net_offsets[name]
            target_poly = Polygon(
                [offset[0], offset[1], 0],
                [offset[0]+1, offset[1], 0],
                [offset[0]+1, offset[1]+1, 0],
                [offset[0], offset[1]+1, 0],
                color=WHITE, stroke_width=2, fill_opacity=0.1
            )
            target_faces.add(target_poly)
        
        # Position the net (Issue 38 Fix)
        self.place_in_area(target_faces, "C2", "F5", scale_factor=0.7)

        # Map dots to new positions on the net using vertices of the positioned polygons
        spider_target_pos = target_faces[0].get_vertices()[0] # Bottom (0,-1) corner
        fly_target_pos = target_faces[1].get_vertices()[2]    # Front (1,1) corner

        self.play(
            ReplacementTransform(faces, target_faces),
            spider_dot.animate.move_to(spider_target_pos),
            fly_dot.animate.move_to(fly_target_pos),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        shortest_path_net = Line(spider_target_pos, fly_target_pos, color="#FFFF00", stroke_width=4)
        self.play(Create(shortest_path_net))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Folding back: Map the line segments back to the cube faces
        final_faces = VGroup()
        for face in face_data:
            poly = Polygon(*[project_3d(v) for v in face["verts"]], 
                           color=WHITE, stroke_width=2, fill_opacity=0.1)
            final_faces.add(poly)
            
        # Define path segments in 3D projection
        p_start_raw = [0,0,0]
        p_mid_raw = [0.5, 0, 1]
        p_end_raw = [1,1,1]
        
        folded_path = VGroup(
            Line(project_3d(p_start_raw), project_3d(p_mid_raw), color="#FFFF00", stroke_width=4),
            Line(project_3d(p_mid_raw), project_3d(p_end_raw), color="#FFFF00", stroke_width=4)
        )
        
        # New dot objects for better transition alignment
        s_dot_final = Dot(project_3d(p_start_raw), color="#FF0000", radius=0.08)
        f_dot_final = Dot(project_3d(p_end_raw), color="#0000FF", radius=0.08)

        # Position final group (Issue 40 Fix)
        final_group = VGroup(final_faces, folded_path, s_dot_final, f_dot_final)
        self.place_in_area(final_group, "D3", "F6", scale_factor=1.0)

        self.play(
            ReplacementTransform(target_faces, final_faces),
            ReplacementTransform(shortest_path_net, folded_path),
            ReplacementTransform(spider_dot, s_dot_final),
            ReplacementTransform(fly_dot, f_dot_final),
            run_time=2
        )
        self.wait(2)
