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
        # Setup layout with title and lecture lines
        # As per mandatory instruction #4
        title_text = "Prerequisite: The Jordan Curve"
        lecture_lines = [
            "We call this loop a Jordan curve.",
            "It is continuous and never intersects itself.",
            "Think of a simple, unbroken rubber band."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1 in #00FFFF (matches shape color)
        self.play(self.lecture[0].animate.set_color("#00FFFF"))

        # Initial shape: Circle in #00FFFF
        circle = Circle(radius=1.5, color="#00FFFF")
        # ISSUE 43 FIX: Positioned in B2-D5 area
        self.place_in_area(circle, "B2", "D5", scale_factor=0.8)
        
        # Second shape: Star in #00FFFF
        star = Star(n=7, inner_radius=0.8, outer_radius=1.5, color="#00FFFF")
        # ISSUE 43 FIX: Positioned in B2-D5 area
        self.place_in_area(star, "B2", "D5", scale_factor=0.8)

        # Third shape: Amoeba (smooth blob) in #00FFFF
        amoeba_pts = [
            [1.5, 0.5, 0], [0.8, 1.5, 0], [-0.5, 1.2, 0], [-1.5, 0.8, 0],
            [-1.2, -0.5, 0], [-0.5, -1.5, 0], [0.8, -1.2, 0], [1.6, -0.8, 0]
        ]
        amoeba = VMobject(color="#00FFFF")
        amoeba.set_points_as_corners([*amoeba_pts, amoeba_pts[0]])
        amoeba.make_smooth()
        # ISSUE 43 FIX: Positioned in B2-D5 area
        self.place_in_area(amoeba, "B2", "D5", scale_factor=0.8)

        # Sequence of morphs
        self.play(Create(circle))
        self.wait(0.5)
        self.play(ReplacementTransform(circle, star))
        self.wait(0.5)
        self.play(ReplacementTransform(star, amoeba))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2 in #FFFFFF (matches flash color)
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFFFF")
        )

        # Flash a small section of the curve to show continuity
        # Create a thick duplicate of the curve to flash
        flash_path = amoeba.copy().set_stroke(color="#FFFFFF", width=12)
        
        # Passing flash animation along the curve
        self.play(ShowPassingFlash(flash_path, time_width=0.3, run_time=2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3 in #FFFFFF (matches label color)
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFFFF")
        )

        # Label "Jordan Curve: No self-intersections" appears in #FFFFFF
        label = Text("Jordan Curve:\nNo self-intersections", font_size=24, color="#FFFFFF")
        # ISSUE 42 FIX: Placed in area E2-F5 for better vertical alignment and spacing
        self.place_in_area(label, "E2", "F5", scale_factor=0.8)
        
        self.play(Write(label))
        self.wait(2)
