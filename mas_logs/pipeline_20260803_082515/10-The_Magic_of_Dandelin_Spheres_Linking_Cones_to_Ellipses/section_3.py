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
        # Title and Lecture Lines
        title_str = "Enter the Dandelin Spheres"
        lecture_lines = [
            "We place two spheres inside the cone.",
            "Each touches the cone and the slicing plane.",
            "The contact points on the plane are the foci."
        ]
        self.setup_layout(title_str, lecture_lines)

        # Colors
        CONE_COLOR = "#FFD700"
        PLANE_COLOR = "#ADD8E6"
        LIME_SPHERE = "#7FFF00"
        CYAN_SPHERE = "#00FFFF"
        FOCI_COLOR = "#FF00FF"

        # Asset Paths
        CONE_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg"
        SPHERE_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg"

        # === Animation for Lecture Line 1 ===
        # Highlight Lecture Line 1
        self.play(self.lecture[0].animate.set_color(CONE_COLOR))

        # Cone SVG
        cone = SVGMobject(CONE_ASSET).set_color(CONE_COLOR)
        
        # Tilted Plane Line (2D representation)
        # Using specific grid points to create a tilted line across the cone's path
        plane_line = Line(self.grid["B3"], self.grid["E5"], color=PLANE_COLOR)

        # Geometry Group
        geometry_base = VGroup(cone, plane_line)
        # Issue 20: Place geometry_base in area A3 to F6
        self.place_in_area(geometry_base, "A3", "F6", scale_factor=1.2)

        self.play(Create(cone), Create(plane_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Lecture Line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(LIME_SPHERE)
        )

        # Small Lime Sphere (SVG)
        small_sphere = SVGMobject(SPHERE_ASSET).set_color(LIME_SPHERE).set_opacity(0.6)
        # Position small sphere above the plane, tangent to cone and plane
        self.place_at_grid(small_sphere, "C4", scale_factor=0.6)
        
        # Large Cyan Sphere (SVG)
        large_sphere = SVGMobject(SPHERE_ASSET).set_color(CYAN_SPHERE).set_opacity(0.6)
        # Position large sphere below the plane, tangent to cone and plane
        self.place_at_grid(large_sphere, "D5", scale_factor=1.1)

        # Animations for spheres expanding
        self.play(small_sphere.animate.scale(1.2), run_time=1.5) # Expand effect
        self.wait(0.5)
        self.play(
            self.lecture[1].animate.set_color(CYAN_SPHERE),
            large_sphere.animate.scale(1.1), # Expand effect
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight Lecture Line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(FOCI_COLOR)
        )

        # Contact points on the plane
        # F1 corresponds to small sphere contact
        f1 = Dot(color=FOCI_COLOR)
        self.place_at_grid(f1, "C4", scale_factor=0.8)
        f1.shift(RIGHT * 0.2 + DOWN * 0.2) # Offset to plane line
        
        # F2 corresponds to large sphere contact
        f2 = Dot(color=FOCI_COLOR)
        self.place_at_grid(f2, "D5", scale_factor=0.8)
        f2.shift(LEFT * 0.3 + UP * 0.3) # Offset to plane line
        
        # Foci Labels
        f1_label = Text("F1", font_size=20, color=FOCI_COLOR)
        f2_label = Text("F2", font_size=20, color=FOCI_COLOR)

        # Issue 21: Place F1 label at C3
        self.place_at_grid(f1_label, "C3", scale_factor=0.8)
        # Issue 22: Place F2 label at D5
        self.place_at_grid(f2_label, "D5", scale_factor=0.8)
        f2_label.shift(RIGHT * 0.5) # Nudge label so it doesn't overlap Dot F2

        self.play(
            Create(f1), Create(f2),
            Write(f1_label), Write(f2_label)
        )
        self.wait(2)

        # Cleanup: Return lecture color to white
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
