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
        # Data
        title_text = "Wide Matrices: Downgrading Dimensions (3D to 2D)"
        lecture_lines = [
            "A two-by-three matrix squishes three-D into two-D.",
            "This acts like a flashlight casting a shadow.",
            "Imagine a three-D bird flying in the sky.",
            "The matrix projects its shadow onto the ground.",
            "Information is lost as volume collapses into area."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        CUBE_COLOR = "#0000FF"
        PLANE_COLOR = "#FFFFFF"
        SHADOW_COLOR = "#444444"
        LINE_COLOR = "#AAAAAA"
        
        # Helper for Wireframe Cube
        def get_cube_mobject(color):
            # 8 vertices
            # Front face
            v = [
                np.array([-0.5, 0.5, 0]), np.array([0.5, 0.5, 0]), 
                np.array([0.5, -0.5, 0]), np.array([-0.5, -0.5, 0]),
                # Back face (shifted for 3D perspective)
                np.array([-0.2, 0.8, 0]), np.array([0.8, 0.8, 0]), 
                np.array([0.8, -0.2, 0]), np.array([-0.2, -0.2, 0])
            ]
            edges = [
                (0,1), (1,2), (2,3), (3,0), # Front edges [0:4]
                (4,5), (5,6), (6,7), (7,4), # Back edges [4:8]
                (0,4), (1,5), (2,6), (3,7)  # Connection edges [8:12]
            ]
            vg = VGroup()
            for e in edges:
                vg.add(Line(v[e[0]], v[e[1]], color=color, stroke_width=2))
            return vg

        # === Animation for Lecture Line 1 ===
        # A two-by-three matrix squishes three-D into two-D.
        self.lecture[0].set_color(CUBE_COLOR)
        
        cube = get_cube_mobject(CUBE_COLOR)
        # Resolved Issue #28, #29: Move cube to B3 and reduce scale to 0.8
        self.place_at_grid(cube, "B3", scale_factor=0.8)
        
        # Create Plane (Parallelogram)
        plane = Polygon(
            np.array([-1.5, 0.4, 0]), np.array([1.5, 0.4, 0]), 
            np.array([1.0, -0.4, 0]), np.array([-2.0, -0.4, 0]),
            color=PLANE_COLOR, fill_opacity=0.2, stroke_width=2
        )
        # Resolved Issue #30: Scale plane to 1.2 at E3
        self.place_at_grid(plane, "E3", scale_factor=1.2)
        
        self.play(FadeIn(cube), FadeIn(plane))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # This acts like a flashlight casting a shadow.
        self.lecture[1].set_color(WHITE)
        self.play(Indicate(cube, color=CUBE_COLOR))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Imagine a three-D bird flying in the sky.
        self.lecture[2].set_color(CUBE_COLOR)
        # Small vertical hovering to represent "bird"
        self.play(cube.animate.shift(UP * 0.2), run_time=0.8)
        self.play(cube.animate.shift(DOWN * 0.2), run_time=0.8)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The matrix projects its shadow onto the ground.
        self.lecture[3].set_color(SHADOW_COLOR)
        
        # Calculate current positions for vertices based on line segments
        # Vertices 0-3 are start points of edges 0-3. Vertices 4-7 are start points of edges 4-7.
        global_v = [cube[i].get_start() for i in range(8)]
        
        # Target Y coordinate is the plane's vertical center
        plane_y = plane.get_center()[1]
        
        shadow_lines = VGroup()
        shadow_vertices = []
        for v_glob in global_v:
            # Projection lines
            end_p = np.array([v_glob[0], plane_y + (v_glob[1] - cube.get_center()[1])*0.3, 0])
            # The shadow itself needs to be "flat", so we adjust the Y-spread of the shadow vertices
            # to be smaller than the cube's vertical spread to look like a projection on a tilted plane.
            shadow_lines.add(DashedLine(v_glob, end_p, color=LINE_COLOR, stroke_width=1, dash_length=0.05))
            shadow_vertices.append(end_p)
            
        # The 2D shadow wireframe
        shadow_cube = VGroup()
        edges = [
            (0,1), (1,2), (2,3), (3,0), # Front
            (4,5), (5,6), (6,7), (7,4), # Back
            (0,4), (1,5), (2,6), (3,7)  # Connections
        ]
        for e in edges:
            shadow_cube.add(Line(shadow_vertices[e[0]], shadow_vertices[e[1]], color=SHADOW_COLOR, stroke_width=2))
            
        self.play(Create(shadow_lines), run_time=1.5)
        self.play(FadeIn(shadow_cube))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # Information is lost as volume collapses into area.
        self.lecture[4].set_color(SHADOW_COLOR)
        
        # Fade out cube and projection lines, leave flattened shadow
        self.play(
            FadeOut(cube),
            FadeOut(shadow_lines),
            shadow_cube.animate.set_stroke(opacity=0.6)
        )
        self.play(Indicate(shadow_cube, color=SHADOW_COLOR))
        self.wait(2)
