from manim import *
import numpy as np
import os

# Pre-emptively create the media/texts directory
os.makedirs(os.path.join("media", "texts"), exist_ok=True)

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
        # Colors
        CONE_COLOR = "#FFA500"
        PLANE_COLOR = "#00FF00"
        SECTION_COLOR = "#FFFFFF"
        CONE_ASSET_PATH = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cone.svg"

        # Setup the layout
        self.setup_layout(
            "The Slicing Puzzle: Cross-Sectional Logic",
            [
                "Slicing reveals the internal shapes hidden within three dimensions.",
                "A flat plane cuts horizontally through our solid cone.",
                "This creates a circle where the two surfaces intersect.",
                "Tilting the plane produces an elongated elliptical shape.",
                "Slicing through the tip exposes a sharp triangle.",
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(CONE_COLOR))
        
        # Load SVG Cone [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cone.svg]
        # Resolve Issue 22: Integrate asset
        cone = SVGMobject(CONE_ASSET_PATH)
        cone.set_color(CONE_COLOR)
        cone.set_stroke(width=2)
        cone.set_fill(CONE_COLOR, opacity=0.4)
        
        # Resolve Issue 29 & 30: Place in C3-F5 area with scale 0.8
        self.place_in_area(cone, "C3", "F5", scale_factor=0.8)
        
        self.play(FadeIn(cone))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(PLANE_COLOR)
        )
        
        # Horizontal Plane representation
        # Adjusted size to not extend too far (Issue 30)
        plane_width = 2.0
        plane_height = 0.5
        plane = Polygon(
            [-plane_width/2, plane_height/2, 0], 
            [plane_width/2, plane_height/2, 0], 
            [plane_width/2 + 0.3, -plane_height/2, 0], 
            [-plane_width/2 - 0.3, -plane_height/2, 0],
            color=PLANE_COLOR, stroke_width=2
        ).set_fill(PLANE_COLOR, opacity=0.3)
        
        # Position plane within the cone height
        # Using cone height to position
        plane.move_to(cone.get_center() + UP * 0.1)
        
        self.play(FadeIn(plane))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(SECTION_COLOR)
        )
        
        # Circle cross-section (visually an ellipse in 2D projection)
        circle_section = Ellipse(width=0.8, height=0.2, color=SECTION_COLOR, stroke_width=4)
        circle_section.move_to(plane.get_center())
        
        self.play(Create(circle_section))
        self.play(Flash(circle_section, color=SECTION_COLOR))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(SECTION_COLOR)
        )
        
        # Tilted Plane
        tilted_plane = plane.copy().rotate(-15 * DEGREES)
        tilted_plane.move_to(cone.get_center())
        
        # Ellipse cross-section
        ellipse_section = Ellipse(width=1.2, height=0.3, color=SECTION_COLOR, stroke_width=4)
        ellipse_section.rotate(-15 * DEGREES)
        ellipse_section.move_to(cone.get_center())
        
        self.play(
            Transform(plane, tilted_plane),
            Transform(circle_section, ellipse_section)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(SECTION_COLOR)
        )
        
        # Vertical Plane through tip
        # Adjusted height to stay within grid/cone bounds (Issue 30)
        vertical_plane = Rectangle(width=0.1, height=2.0, color=PLANE_COLOR, stroke_width=2).set_fill(PLANE_COLOR, opacity=0.3)
        vertical_plane.apply_matrix(np.array([[1, 0, 0], [0.1, 1, 0], [0, 0, 1]]))
        vertical_plane.move_to(cone.get_center())
        
        # Triangle cross-section
        # Calculate vertices relative to the cone object
        cone_height = cone.get_height()
        cone_width = cone.get_width()
        apex = cone.get_top()
        left_corner = cone.get_bottom() + LEFT * (cone_width / 2.2)
        right_corner = cone.get_bottom() + RIGHT * (cone_width / 2.2)
        
        triangle_section = Polygon(
            apex, left_corner, right_corner,
            color=SECTION_COLOR, stroke_width=4
        )
        
        self.play(
            FadeOut(plane),
            FadeIn(vertical_plane),
            Transform(circle_section, triangle_section)
        )
        self.play(Flash(triangle_section, color=SECTION_COLOR))
        self.wait(2)

        # Cleanup highlight
        self.play(self.lecture[4].animate.set_color(WHITE))
