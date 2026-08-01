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
        # Title and Lecture Lines from Storyboard
        title_str = "The Wide Matrix: The Great Squish (3x2 Transform)"
        lecture_lines = [
            "A wide 2x3 matrix squashes 3D into 2D space.",
            "Three input coordinates are projected onto a flat plane.",
            "Depth information disappears during this dimensional reduction."
        ]
        self.setup_layout(title_str, lecture_lines)

        # Define Colors
        COLOR_BIRD = WHITE
        COLOR_MATRIX = "#FF00FF" # Magenta as per storyboard
        COLOR_SHADOW = "#808080" # Grey
        COLOR_GROUND = BLUE_E

        # Projection helper (Simulated Isometric Perspective)
        def p3(p):
            x, y, z = p
            return np.array([
                (x - y) * 0.8,
                (x + y) * 0.4 + z * 0.6,
                0
            ])

        # === 3D Visualization Setup ===
        three_d_container = VGroup()
        
        # 1. Coordinate Axes
        origin_p = [0, 0, 0]
        axes_lines = VGroup(
            Line(p3(origin_p), p3([2, 0, 0]), color=WHITE, stroke_width=2).add_tip(tip_length=0.15),
            Line(p3(origin_p), p3([0, 2, 0]), color=WHITE, stroke_width=2).add_tip(tip_length=0.15),
            Line(p3(origin_p), p3([0, 0, 2]), color=WHITE, stroke_width=2).add_tip(tip_length=0.15)
        )
        labels = VGroup(
            MathTex("x", font_size=16).move_to(p3([2.3, 0, 0])),
            MathTex("y", font_size=16).move_to(p3([0, 2.3, 0])),
            MathTex("z", font_size=16).move_to(p3([0, 0, 2.3]))
        )
        axes = VGroup(axes_lines, labels)
        
        # 2. Ground Plane (the XY-plane projection target)
        ground = Polygon(
            p3([-1.8, -1.8, 0]), p3([1.8, -1.8, 0]), 
            p3([1.8, 1.8, 0]), p3([-1.8, 1.8, 0]),
            fill_color=COLOR_GROUND, fill_opacity=0.15, stroke_color=COLOR_GROUND, stroke_width=1
        )
        
        # 3. Bird Model [Asset: bird.svg]
        bird = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bird.svg")
        bird.set_color(COLOR_BIRD)
        bird.scale(0.3)
        bird.move_to(p3([0, 0, 1.4]))
        
        # 4. Shadow [Asset: bird.svg] (Projected onto z=0)
        shadow = bird.copy()
        shadow.set_color(COLOR_SHADOW)
        shadow.set_opacity(0.5)
        # Flatten the shadow vertically to match the projection plane look
        shadow.stretch(0.5, dim=1)
        shadow.move_to(p3([0, 0, 0]))
        
        # 5. Projection Lines (Visualizing the squish)
        # Using a few points from the bird's bounding box for simplicity
        proj_lines = VGroup(*[
            DashedLine(bird.get_corner(dir), shadow.get_corner(dir), color=COLOR_SHADOW, stroke_width=1, dash_length=0.05)
            for dir in [UL, UR, DL, DR]
        ])

        # Group components into the container
        three_d_container.add(ground, axes, bird, shadow, proj_lines)
        
        # Initially hide shadow and lines
        shadow.set_fill(opacity=0).set_stroke(opacity=0)
        proj_lines.set_stroke(opacity=0)
        
        # Fix: Positioning and scaling per VideoCritic suggestions (Issues 26, 28)
        self.place_in_area(three_d_container, 'B2', 'F6', scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        self.play(Create(axes), FadeIn(ground), run_time=1.5)
        self.play(FadeIn(bird), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_MATRIX)
        
        mat_obj = Matrix([[1, 0, 0], [0, 1, 0]], 
                        left_bracket="[", right_bracket="]",
                        element_to_mobject_config={"color": COLOR_MATRIX}).set_color(COLOR_MATRIX)
        mat_label = MathTex("A = ", color=COLOR_MATRIX, font_size=32)
        matrix_full = VGroup(mat_label, mat_obj).arrange(RIGHT, buff=0.15)
        
        # Fix: Centering matrix per VideoCritic suggestion (Issue 27)
        self.place_in_area(matrix_full, 'A3', 'A5', scale_factor=0.8)
        
        self.play(Write(matrix_full))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_SHADOW)
        
        self.play(
            shadow.animate.set_fill(opacity=0.5).set_stroke(opacity=0.5),
            Create(proj_lines),
            run_time=2
        )
        # Briefly highlight the loss of 'z' dimension
        self.play(bird.animate.set_fill(opacity=0.3).set_stroke(opacity=0.3), run_time=1)
        self.wait(3)
