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

class Section6Scene(TeachingScene):
    def construct(self):
        # Configuration
        COLOR_FLOW = "#FFFF00"  # Yellow
        COLOR_CAMERA = "#00FFFF" # Cyan
        COLOR_APP = "#FFD700"    # Gold
        
        lines = [
            'Basis, Matrix, Transformation: that is the core flow.',
            'This math powers camera rotations in video games.',
            'Change of basis helps find patterns in complex data.'
        ]
        
        self.setup_layout("Visual Summary & Application", lines)

        # === Animation for Lecture Line 1 ===
        # Flow diagram: Basis -> Matrix P -> Transformation
        self.play(self.lecture[0].animate.set_color(COLOR_FLOW))
        
        basis_rect = Rectangle(width=1.5, height=0.8, color=COLOR_FLOW)
        basis_text = Text("Basis", font_size=20, color=COLOR_FLOW)
        basis_grp = VGroup(basis_rect, basis_text)
        self.place_at_grid(basis_grp, "B1", scale_factor=0.8)
        
        matrix_rect = Rectangle(width=1.5, height=0.8, color=COLOR_FLOW)
        matrix_text = Text("Matrix P", font_size=20, color=COLOR_FLOW)
        matrix_grp = VGroup(matrix_rect, matrix_text)
        self.place_at_grid(matrix_grp, "B3", scale_factor=0.8)
        
        trans_rect = Rectangle(width=2.0, height=0.8, color=COLOR_FLOW)
        trans_text = Text("Transformation", font_size=18, color=COLOR_FLOW)
        trans_grp = VGroup(trans_rect, trans_text)
        # Fix: Issue 44/50 - Reduced gap in flow
        self.place_at_grid(trans_grp, "B4", scale_factor=0.6) # Reduced scale slightly to avoid overlap at B4
        
        arrow1 = Arrow(start=basis_grp.get_right(), end=matrix_grp.get_left(), color=COLOR_FLOW, buff=0.1)
        arrow2 = Arrow(start=matrix_grp.get_right(), end=trans_grp.get_left(), color=COLOR_FLOW, buff=0.1)
        
        self.play(
            FadeIn(basis_grp),
            Create(arrow1),
            FadeIn(matrix_grp),
            Create(arrow2),
            FadeIn(trans_grp)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Load SVG Asset: Issue 27/50
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_CAMERA)
        )
        
        # Camera icon asset integration
        camera_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/camera.svg")
        camera_icon.set_color(COLOR_CAMERA)
        
        # Local Basis Vectors for Camera context
        v1 = Arrow(start=ORIGIN, end=RIGHT*0.8, color=RED, buff=0)
        v2 = Arrow(start=ORIGIN, end=UP*0.8, color=GREEN, buff=0)
        camera_basis = VGroup(v1, v2)
        
        camera_system = VGroup(camera_icon, camera_basis)
        self.place_in_area(camera_system, "D1", "D3", scale_factor=0.8)
        
        self.play(FadeIn(camera_system))
        self.play(camera_system.animate.rotate(45 * DEGREES), run_time=2)
        self.play(camera_system.animate.rotate(-90 * DEGREES), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show Bob and Z-4 together at the gold star's location
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_APP)
        )
        
        # Issue 42/50: Star at E5
        star = Star(n=5, color=COLOR_APP, fill_opacity=1)
        self.place_at_grid(star, "E5", scale_factor=0.8)
        
        # Bob
        bob = VGroup(Circle(radius=0.15, color=WHITE), Line(DOWN*0.15, DOWN*0.5, color=WHITE))
        self.place_at_grid(bob, "E4", scale_factor=1.0)
        bob_label = Text("Bob", font_size=16, color=WHITE).next_to(bob, UP, buff=0.1)
        
        # Issue 43/50: Z-4 at E6
        z4 = VGroup(Square(side_length=0.2, color=WHITE), Square(side_length=0.3, color=WHITE).next_to(ORIGIN, DOWN, buff=0))
        self.place_at_grid(z4, "E6", scale_factor=0.8)
        z4_label = Text("Z-4", font_size=16, color=WHITE).next_to(z4, UP, buff=0.1)
        
        self.play(
            FadeIn(star),
            FadeIn(bob), FadeIn(bob_label),
            FadeIn(z4), FadeIn(z4_label)
        )
        
        # Final Move: characters move to focus on the star
        star_center = self.grid["E5"]
        self.play(
            bob.animate.move_to(star_center + LEFT*0.6),
            bob_label.animate.move_to(star_center + LEFT*0.6 + UP*0.4),
            z4.animate.move_to(star_center + RIGHT*0.6),
            z4_label.animate.move_to(star_center + RIGHT*0.6 + UP*0.4),
            star.animate.scale(1.2),
            run_time=2
        )
        
        self.wait(3)
