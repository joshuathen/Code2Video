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
        # Setup the layout with title and lecture lines
        self.setup_layout(
            "Summary: Linear Transformations as Dimensional Portals", 
            [
                "Non-square matrices act as portals between different dimensions.", 
                "They map data across varying levels of reality.", 
                "Visualizing these shifts reveals the power of linear algebra."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Show a 2D shape (#FFFFFF) transitioning into a 3D surface (#00FF00) via a tall matrix.
        
        # Create 2D shape (White circle)
        shape_2d = Circle(radius=1.2, color="#FFFFFF", stroke_width=4, fill_opacity=0.2)
        self.place_in_area(shape_2d, "A2", "C5")
        
        # Build a 3D-looking projected surface (Green)
        surface_3d = VGroup()
        # Skewed vertices to simulate 3D perspective in a 2D Scene
        v = [
            np.array([-1.5, -0.6, 0]), np.array([0.5, -1.4, 0]), 
            np.array([1.5, 1.0, 0]), np.array([-0.5, 1.8, 0])
        ]
        base_plane = Polygon(*v, color="#00FF00", fill_opacity=0.5, stroke_width=2)
        surface_3d.add(base_plane)
        
        # Add grid lines for "surface" feel
        for i in range(1, 4):
            frac = i / 4
            surface_3d.add(Line(interpolate(v[0], v[1], frac), interpolate(v[3], v[2], frac), color="#00FF00", stroke_opacity=0.4))
            surface_3d.add(Line(interpolate(v[0], v[3], frac), interpolate(v[1], v[2], frac), color="#00FF00", stroke_opacity=0.4))
        
        self.place_in_area(surface_3d, "A2", "C5")
        
        # Initial display
        self.play(Create(shape_2d), run_time=1)
        self.wait(0.5)
        
        # Transition to 3D surface and highlight Line 1 in Green
        self.play(
            ReplacementTransform(shape_2d, surface_3d),
            self.lecture[0].animate.set_color("#00FF00"),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Squash that 3D surface into a 1D line segment (#FF0000) via a wide matrix.
        
        # Create 1D line segment (Red)
        line_1d = Line(LEFT * 2, RIGHT * 2, color="#FF0000", stroke_width=10)
        self.place_in_area(line_1d, "D2", "D5")
        
        # Transform surface to line and highlight Line 2 in Red
        self.play(
            ReplacementTransform(surface_3d, line_1d),
            self.lecture[1].animate.set_color("#FF0000"),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display the concluding text 'Linear Transformations: Dimensional Portals' (#FFFFFF).
        
        concluding_text = Text(
            "Linear Transformations:\nDimensional Portals", 
            font_size=28, 
            color="#FFFFFF"
        )
        self.place_in_area(concluding_text, "E2", "F5")
        
        # Show text and highlight Line 3 in White (already white, but emphasizing color adherence)
        self.play(
            Write(concluding_text),
            self.lecture[2].animate.set_color("#FFFFFF"),
            run_time=2
        )
        self.wait(3)
