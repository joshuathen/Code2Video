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
        # Setup the layout with the title and lecture lines
        lecture_lines = [
            'Welcome to the world of linear transformations and space.',
            'Meet Pixel, our square cat residing on a grid.',
            'Watch as a matrix stretches his world outwards.',
            "Pixel's area has increased significantly after this transformation.",
            'The determinant measures this exact change in area.'
        ]
        self.setup_layout("Introduction: The Transformation Factor", lecture_lines)

        # Remove title initially to fade it in with the first animation
        self.remove(self.title)

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))

        # Create a coordinate grid and visual group
        grid_lines = VGroup()
        for x in range(-1, 5):
            grid_lines.add(Line([x, -1, 0], [x, 4, 0], color=GREY, stroke_width=1, stroke_opacity=0.5))
        for y in range(-1, 5):
            grid_lines.add(Line([-1, y, 0], [4, y, 0], color=GREY, stroke_width=1, stroke_opacity=0.5))
        
        # Pixel the Cat [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/concrete.svg]
        pixel = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/concrete.svg")
        pixel.set_color("#00FFFF")
        pixel.stretch_to_fit_width(1)
        pixel.stretch_to_fit_height(1)
        # Position Pixel so its bottom-left corner is at coordinate (0,0)
        pixel.move_to([0.5, 0.5, 0])
        
        # A tracker point to identify the grid origin (0,0) for the transformation
        origin_tracker = Dot(point=[0, 0, 0], fill_opacity=0)
        
        # Group components together
        visual_group = VGroup(grid_lines, pixel, origin_tracker)
        
        # Initially hide Pixel
        pixel.set_opacity(0)
        
        # Position the group in the lower part of the 6x6 workspace (C1 to F6)
        self.place_in_area(visual_group, "C1", "F6", scale_factor=0.6)
        
        # Fade in the grid and the title
        self.play(FadeIn(grid_lines), FadeIn(self.title))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FFFF")
        )
        # Reveal Pixel
        self.play(pixel.animate.set_opacity(1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FFFF")
        )

        # Matrix transformation: Stretch x by 3 and y by 2 (Area becomes 1*3*2 = 6)
        transformation_matrix = np.array([
            [3, 0, 0],
            [0, 2, 0],
            [0, 0, 1]
        ])

        # Animate the stretch around the origin tracker's center
        self.play(
            visual_group.animate.apply_matrix(
                transformation_matrix, 
                about_point=origin_tracker.get_center()
            ),
            run_time=2.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight Line 4
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#FFFF00")
        )

        # Highlight the transformed area in yellow
        self.play(
            pixel.animate.set_color("#FFFF00"),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight Line 5
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color("#00FFFF")
        )

        # Labels for the transformation
        factor_label = Text("Area Scale Factor: 6", font_size=24, color=WHITE)
        det_label = Text("det(A) = 6", font_size=24, color=WHITE)
        
        # Positioning labels in Row B to stay within 1 grid unit of the visual group
        # Issue 32, 33, 34 resolved here
        self.place_in_area(factor_label, 'B1', 'B3', scale_factor=0.7)
        self.place_in_area(det_label, 'B4', 'B6', scale_factor=0.7)

        self.play(
            Write(factor_label),
            Write(det_label)
        )
        self.wait(3)
