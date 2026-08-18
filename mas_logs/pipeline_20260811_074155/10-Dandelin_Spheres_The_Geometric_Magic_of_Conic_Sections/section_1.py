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
        # Data from storyboard
        title = "The Mystery of the Slice"
        lines = [
            "A cone's intersection with a plane creates beautiful curves.",
            "Slanted cuts specifically produce a shape called an ellipse.",
            "We will prove why this geometry works using spheres."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_CONE = "#FFFFFF"
        COLOR_PLANE = "#1E90FF"
        COLOR_ELLIPSE = "#FFFF00"
        
        # === Animation for Lecture Line 1 ===
        # Show a 3D-looking cone (2D triangle with curved top) and a tilted plane.
        self.lecture[0].set_color(WHITE)
        
        # Cone (vertex at bottom)
        cone_body = Polygon(
            [0, -1.5, 0], [1.2, 1.5, 0], [-1.2, 1.5, 0],
            color=COLOR_CONE, fill_opacity=0.2
        )
        cone_top = Ellipse(width=2.4, height=0.6, color=COLOR_CONE).move_to([0, 1.5, 0])
        cone = VGroup(cone_body, cone_top)
        
        # Tilted Plane (represented as a thin rectangle/parallelogram)
        plane = Polygon(
            [-1.5, 0.5, 0], [1.5, -0.5, 0], [1.5, -0.7, 0], [-1.5, 0.3, 0],
            color=COLOR_PLANE, fill_opacity=0.5
        )
        # Shift plane to intersect cone
        plane.move_to([0, 0.2, 0])
        
        visuals_1 = VGroup(cone, plane)
        # Fix requested in Issue 18: use A4-F6 and scale 0.8
        self.place_in_area(visuals_1, "A4", "F6", scale_factor=0.8)
        
        self.play(FadeIn(cone), run_time=1)
        self.play(Create(plane), run_time=1)
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Highlight the intersection curve (ellipse) in bright yellow.
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(COLOR_ELLIPSE),
            run_time=0.5
        )
        
        # Ellipse at intersection
        # It should be tilted and centered where plane meets cone
        # Adjusted size slightly to account for the 0.8 scaling of visuals_1
        intersection_ellipse = Ellipse(
            width=1.4, height=0.4, color=COLOR_ELLIPSE, stroke_width=6
        )
        intersection_ellipse.rotate(-15 * DEGREES) # Match plane tilt roughly
        intersection_ellipse.move_to(plane.get_center())
        
        self.play(Create(intersection_ellipse), run_time=1)
        self.wait(2)
        
        # === Animation for Lecture Line 3 ===
        # Rotate the view to show the ellipse from a top-down perspective, pulse it, then zoom out.
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(COLOR_ELLIPSE),
            run_time=0.5
        )
        
        # Top-down view simulation: transform ellipse to a circle and fade background
        top_down_ellipse = Ellipse(width=1.6, height=1.6, color=COLOR_ELLIPSE, stroke_width=6)
        top_down_ellipse.move_to(intersection_ellipse.get_center())
        
        self.play(
            FadeOut(cone),
            FadeOut(plane),
            Transform(intersection_ellipse, top_down_ellipse),
            run_time=2
        )
        
        # Pulse the yellow ellipse
        self.play(
            intersection_ellipse.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=1
        )
        self.play(
            intersection_ellipse.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=1
        )
        
        # Zoom out/Return to perspective: Bring back cone/plane and transform ellipse back
        original_ellipse = Ellipse(
            width=1.4, height=0.4, color=COLOR_ELLIPSE, stroke_width=6
        ).rotate(-15 * DEGREES).move_to(plane.get_center())
        
        self.play(
            FadeIn(cone),
            FadeIn(plane),
            Transform(intersection_ellipse, original_ellipse),
            run_time=2
        )
        
        self.wait(2)
        
        # Reset colors for final state
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(WHITE),
            run_time=1
        )
        self.wait(2)
