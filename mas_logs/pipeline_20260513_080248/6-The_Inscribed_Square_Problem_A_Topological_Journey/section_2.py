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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup the scene with title and lecture lines
        lecture_lines = [
            'A Jordan curve is a simple closed loop.',
            'It never crosses itself and has no gaps.',
            'Any shape can be deformed into this loop.'
        ]
        self.setup_layout("Prerequisite Knowledge: Jordan Curves", lecture_lines)

        # Colors
        CYAN = "#00FFFF"
        YELLOW = "#FFFF00"
        GREEN = "#00FF00"

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(CYAN))
        
        # Draw a cyan circle
        circle = Circle(radius=1.5, color=CYAN)
        # Resolved Issue 35: Lowering position to avoid top-level labels
        self.place_in_area(circle, "C2", "F5")
        self.play(Create(circle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )

        # Create the blob shape
        # We manually define points for an irregular but simple closed loop
        # Note: points are defined relative to current circle position
        blob_pts = [
            circle.point_at_angle(0) * 1.2 + RIGHT * 0.2,
            circle.point_at_angle(PI/4) * 0.8 + UP * 0.3,
            circle.point_at_angle(PI/2) * 1.1,
            circle.point_at_angle(3*PI/4) * 0.7 + LEFT * 0.5,
            circle.point_at_angle(PI) * 1.3,
            circle.point_at_angle(5*PI/4) * 0.9 + DOWN * 0.4,
            circle.point_at_angle(3*PI/2) * 1.0,
            circle.point_at_angle(7*PI/4) * 1.1 + RIGHT * 0.3,
        ]
        blob = VMobject(color=CYAN)
        blob.set_points_smoothly([*blob_pts, blob_pts[0]])
        # Resolved Issue 36: Lowering position to avoid overlap with labels in row B
        self.place_in_area(blob, "C2", "F5")

        # Labels
        jordan_label = Text("Jordan Curve", font_size=24, color=YELLOW)
        self.place_in_area(jordan_label, "A2", "A5")
        
        simple_label = Text("Simple & Closed", font_size=20, color=YELLOW)
        self.place_in_area(simple_label, "B2", "B5")

        self.play(
            Transform(circle, blob),
            FadeIn(jordan_label, shift=UP),
            FadeIn(simple_label, shift=UP)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(GREEN)
        )

        # Animate a secondary deformation to emphasize flexibility
        blob_pts_v2 = [
            p * (1.1 if i % 2 == 0 else 0.9) for i, p in enumerate(blob_pts)
        ]
        blob_v2 = VMobject(color=CYAN)
        blob_v2.set_points_smoothly([*blob_pts_v2, blob_pts_v2[0]])
        # Resolved Issue 37: Lowering position to avoid overlap
        self.place_in_area(blob_v2, "C2", "F5")

        self.play(Transform(circle, blob_v2), run_time=2, rate_func=there_and_back)
        self.wait(2)

        # Final cleanup for the section
        self.play(
            FadeOut(circle),
            FadeOut(jordan_label),
            FadeOut(simple_label),
            self.lecture[2].animate.set_color(WHITE)
        )
        self.wait(1)
