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

class Section1Scene(TeachingScene):
    def construct(self):
        # Data from shared state
        title_str = "The Dimensional Portal Concept"
        lecture_lines = [
            "Square matrices transform space within the same dimension.",
            "Non-square matrices act as portals between different worlds.",
            "Meet Pixel, a cat living in a 2D grid."
        ]
        self.setup_layout(title_str, lecture_lines)

        # Colors
        color_square = "#87CEEB"    # Sky Blue
        color_non_square = "#00FFFF" # Cyan
        color_pixel = "#FFD700"     # Golden Yellow
        color_grid = "#444444"      # Dark Gray

        # === Animation for Lecture Line 1 ===
        # Line: "Square matrices transform space within the same dimension."
        # Anim: Create a 2D grid (#444444) with Pixel (#FFD700) at (1, 1).
        
        self.play(self.lecture[0].animate.set_color(color_square))
        
        # NumberPlane for 2D world
        # Fix 22: self.place_in_area(grid_2d, 'C3', 'F5', scale_factor=0.8)
        grid_2d = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            background_line_style={"stroke_color": color_grid, "stroke_width": 2},
            axis_config={"stroke_color": color_grid}
        )
        self.place_in_area(grid_2d, "C3", "F5", scale_factor=0.8)
        
        # Pixel the Cat (Yellow circle as per storyboard)
        pixel = Circle(radius=0.15, color=color_pixel, fill_opacity=1)
        pixel_label = Text("Pixel", font_size=18, color=color_pixel)
        pixel_label.next_to(pixel, UP, buff=0.1)
        pixel_group = VGroup(pixel, pixel_label)
        
        # Place pixel at (1,1) relative to the grid_2d
        # Use grid_2d.c2p for consistent positioning within the plane
        pixel_group.move_to(grid_2d.c2p(1, 1))

        self.play(Create(grid_2d), FadeIn(pixel_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line: "Non-square matrices act as portals between different worlds."
        # Anim: Transform the 2D grid using a 2x2 rotation matrix; the grid rotates, but Pixel remains within the 2D plane.
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color_non_square)
        )
        
        # Show rotation matrix
        # Fix 21: self.place_in_area(rotation_matrix, 'A4', 'A6', scale_factor=0.6)
        rotation_matrix = MathTex(
            r"R = \begin{bmatrix} \cos(45^\circ) & -\sin(45^\circ) \\ \sin(45^\circ) & \cos(45^\circ) \end{bmatrix}",
            color=color_square,
            font_size=24
        )
        self.place_in_area(rotation_matrix, "A4", "A6", scale_factor=0.6)
        
        self.play(Write(rotation_matrix))
        
        # Rotation animation (45 degrees)
        # Note: Pixel rotates with the grid to show he stays in the same plane
        self.play(
            grid_2d.animate.rotate(45 * DEGREES),
            pixel_group.animate.rotate(45 * DEGREES, about_point=grid_2d.get_center()),
            run_time=2
        )
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Line: "Meet Pixel, a cat living in a 2D grid."
        # Anim: Replace the grid with a glowing rectangular frame (#00FFFF) that pulses, representing a dimensional portal.
        
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color_pixel)
        )
        
        # Replace grid with glowing portal
        # Fix 23: self.place_in_area(portal, 'C3', 'F5', scale_factor=1.0)
        portal = Rectangle(width=3.5, height=3.5, color=color_non_square, stroke_width=6)
        # Use set_stroke for glow simulation
        portal.set_stroke(opacity=0.8)
        self.place_in_area(portal, "C3", "F5", scale_factor=1.0)
        
        # Updater for pulsing effect (L008: use self.renderer.time)
        def portal_pulse(m):
            m.set_stroke(opacity=0.4 + 0.4 * np.sin(self.renderer.time * 4))

        portal.add_updater(portal_pulse)
        
        self.play(
            FadeOut(grid_2d),
            FadeOut(rotation_matrix),
            FadeIn(portal),
            # Transition pixel to the center of the portal
            pixel_group.animate.move_to(portal.get_center())
        )
        # Highlight Pixel as per previous logic (good for emphasizing "Meet Pixel")
        self.play(Indicate(pixel_group, color=color_pixel, scale_factor=1.3))
        self.wait(2)

        # Cleanup
        portal.remove_updater(portal_pulse)
