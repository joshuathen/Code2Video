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

class Section5Scene(ThreeDScene):
    def construct(self):
        # 1. Setup Layout (Title and Lecture Text)
        title_text = "Dandelin Spheres: The 3D Geometry"
        lecture_lines = [
            "1. Spheres touch the cone",
            "   at a circle.",
            "2. They touch the plane",
            "   at a single point.",
            "3. These points are the",
            "   foci of the ellipse."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # 2. 3D Construction Area
        # We will use the right side of the screen for the 3D visualization.
        # Shift the camera to center the right side
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        
        # Define 3D objects
        cone = Cone(base_radius=1.5, height=3, fill_opacity=0.3, color=BLUE)
        cone.shift(RIGHT * 3) # Move to the right side
        
        sphere = Sphere(radius=0.6, color=YELLOW, fill_opacity=0.8)
        sphere.move_to(cone.get_center() + UP * 0.2)
        
        plane = NumberPlane(x_range=[-2, 2], y_range=[-2, 2], background_line_style={"stroke_opacity": 0.4})
        plane.rotate(30 * DEGREES, axis=RIGHT)
        plane.move_to(cone.get_center() + UP * 0.5)
        
        # Labels for 3D objects (using 2D labels projected/fixed)
        sphere_label = Text("Dandelin Sphere", font_size=18).next_to(sphere, RIGHT)

        # 3. Animations
        self.add_fixed_in_frame_mobjects(self.title, self.lecture)
        
        self.play(Create(cone))
        self.play(FadeIn(sphere))
        self.play(Create(plane))
        self.begin_ambient_camera_rotation(rate=0.1)
        self.play(Write(sphere_label))
        self.wait(3)
        self.stop_ambient_camera_rotation()

    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        
        # Left-side lecture content
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.5)
        
        # Define animation grid (6x6) on the right side
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]
        cols = ["1", "2", "3", "4", "5", "6"]

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Offset x to the right side of the screen (approx 0 to 6 range)
                x_val = 1.0 + j * 0.8
                y_val = 2.0 - i * 0.8
                self.grid[f"{row}{col}"] = np.array([x_val, y_val, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        """Helper to place objects in the coordinate grid system."""
        mobject.scale(scale_factor)
        if grid_pos in self.grid:
            mobject.move_to(self.grid[grid_pos])
        return mobject
