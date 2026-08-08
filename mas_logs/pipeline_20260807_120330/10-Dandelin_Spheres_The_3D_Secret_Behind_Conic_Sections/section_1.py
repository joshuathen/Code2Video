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
        # Data for the scene
        title = "The Mystery of the Slice"
        lines = [
            "- Imagine slicing a cone with a flat plane.",
            "- This intersection creates a perfect elliptical curve.",
            "- Why does this slice follow the ellipse formula?"
        ]
        self.setup_layout(title, lines)

        # Colors
        CONE_COLOR = "#888888"
        PLANE_COLOR = "#FFFFFF"
        SLICE_COLOR = "#00FFFF"
        TEXT_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Imagine slicing a cone with a flat plane.
        self.lecture[0].set_color(YELLOW)
        
        # Create a 2D perspective representation of a cone
        cone_base = Ellipse(width=3, height=0.6, color=CONE_COLOR, fill_opacity=0.2)
        cone_sides = Polygon([-1.5, -1.5, 0], [0, 1.5, 0], [1.5, -1.5, 0], color=CONE_COLOR, fill_opacity=0.1)
        cone_base.shift(DOWN * 1.5)
        cone = VGroup(cone_sides, cone_base)
        
        # Applied Issue 37: Scale factor 1.0 and move to A2-D5
        self.place_in_area(cone, "A2", "D5", scale_factor=1.0)
        
        # Create a slicing plane (parallelogram)
        plane = Polygon(
            [-2.5, 0.5, 0], [1.5, 1.5, 0], [2.5, -0.5, 0], [-1.5, -1.5, 0],
            color=PLANE_COLOR, fill_opacity=0.3
        )
        self.place_in_area(plane, "A2", "D5", scale_factor=1.0)
        # Visual adjustment to make it look like it's intersecting the middle
        plane.shift(UP * 0.3)
        
        self.play(FadeIn(cone), FadeIn(plane))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # This intersection creates a perfect elliptical curve.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Tilted ellipse representing the slice
        # Applied Issue 37: Scale factor 1.0 and position at A2-D5
        slice_curve = Ellipse(width=1.6, height=0.6, color=SLICE_COLOR, stroke_width=5)
        slice_curve.rotate(14 * DEGREES)
        self.place_in_area(slice_curve, "A2", "D5", scale_factor=1.0)
        slice_curve.shift(UP * 0.4 + RIGHT * 0.1) # Align with the plane's tilt
        
        self.play(Create(slice_curve))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Why does this slice follow the ellipse formula?
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Simulate camera rotation: transform components to face-on view
        # The slice becomes a standard oval
        final_ellipse = Ellipse(width=3.5, height=2.2, color=SLICE_COLOR, stroke_width=5)
        self.place_in_area(final_ellipse, "B2", "E5", scale_factor=1.0)
        
        # Rotate the view: 
        # 1. Transform the slice to look flat
        # 2. Fade out the cone and plane
        self.play(
            Transform(slice_curve, final_ellipse),
            cone.animate.set_opacity(0),
            plane.animate.set_opacity(0),
            run_time=2
        )
        self.remove(cone, plane)
        
        # Final Question Text (Applied Issue 37: Move to F2-F5)
        question_text = Text("Why is this an Ellipse?", font_size=24, color=TEXT_COLOR)
        self.place_in_area(question_text, "F2", "F5", scale_factor=1.0)
        
        self.play(Write(question_text))
        self.wait(2)
