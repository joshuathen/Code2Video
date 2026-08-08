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

class Section4Scene(TeachingScene):
    def construct(self):
        title_text = "The 'Spiky' Sphere: Sphere vs. Cube"
        lines = [
            "Inscribe an n-sphere inside an n-cube of side 2R.",
            "In low dimensions, the sphere fills the cube.",
            "High dimensions push the cube's corners further away.",
            "The sphere becomes a tiny speck in the center.",
            "Cube corners stretch into incredibly long, thin spikes."
        ]
        self.setup_layout(title_text, lines)

        # Colors
        COLOR_SPHERE = "#00FF00"
        COLOR_CORNER = "#FF0000"
        COLOR_CUBE = WHITE

        # === Animation for Lecture Line 1 ===
        # Inscribe an n-sphere inside an n-cube of side 2R.
        self.lecture[0].set_color(YELLOW)
        
        square = Square(side_length=3.0, color=COLOR_CUBE)
        circle = Circle(radius=1.5, color=COLOR_SPHERE)
        
        geometry_2d = VGroup(square, circle)
        self.place_in_area(geometry_2d, "B3", "E6", scale_factor=1.0)
        
        # Label side 2R
        side_label = MathTex("2R", font_size=24).next_to(square, LEFT, buff=0.2)
        
        self.play(Create(square))
        self.play(Create(circle))
        self.play(Write(side_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # In low dimensions, the sphere fills the cube.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Show circle covers most of the square - highlighting overlap
        self.play(circle.animate.set_fill(COLOR_SPHERE, opacity=0.3))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # High dimensions push the cube's corners further away.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Lines to corners (length R√n)
        corners = square.get_vertices()
        center = square.get_center()
        corner_lines = VGroup(*[Line(center, corner, color=COLOR_CORNER) for corner in corners])
        
        # Distance label (R√2) - resolving Issue 31
        dist_label_2d = MathTex("R\\sqrt{2}", font_size=24, color=COLOR_CORNER)
        self.place_at_grid(dist_label_2d, "B6", scale_factor=0.9)

        self.play(Create(corner_lines), Write(dist_label_2d))
        self.wait(1)

        # Transition to 3D representation to show n growing
        # Simulating 3D: Perspective Cube (Isometric-ish)
        offset = np.array([0.3, 0.3, 0])
        sq_back = Square(side_length=2.5, color=COLOR_CUBE, stroke_opacity=0.5).move_to(center + offset)
        sq_front = Square(side_length=2.5, color=COLOR_CUBE).move_to(center - offset)
        connectors = VGroup(*[
            Line(sq_front.get_vertices()[i], sq_back.get_vertices()[i], color=COLOR_CUBE, stroke_opacity=0.5)
            for i in range(4)
        ])
        cube_3d = VGroup(sq_back, sq_front, connectors)
        
        # Update corner distance label for 3D - resolving Issue 32
        dist_label_3d = MathTex("R\\sqrt{3}", font_size=24, color=COLOR_CORNER)
        self.place_at_grid(dist_label_3d, "B6", scale_factor=0.9)

        self.play(
            FadeOut(geometry_2d), FadeOut(corner_lines), FadeOut(dist_label_2d), FadeOut(side_label),
            FadeIn(cube_3d),
            FadeIn(dist_label_3d)
        )
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # The sphere becomes a tiny speck in the center.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # General Formula - resolving Issue 33
        formula_n = MathTex("R\\sqrt{n}", font_size=32, color=COLOR_CORNER)
        self.place_at_grid(formula_n, "A4", scale_factor=0.8)

        # Sphere as a tiny speck
        sphere_speck = Circle(radius=0.1, color=COLOR_SPHERE, fill_opacity=1).move_to(cube_3d.get_center())
        sphere_label = Text("Vanishing Sphere", font_size=18, color=COLOR_SPHERE)
        self.place_at_grid(sphere_label, "D4", scale_factor=1.0)
        sphere_label.next_to(sphere_speck, DOWN, buff=0.1)

        self.play(
            Write(formula_n),
            FadeIn(sphere_speck),
            Write(sphere_label)
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # Cube corners stretch into incredibly long, thin spikes.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # Hedgehog effect - resolving Issue 22
        # Load asset
        hedgehog = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hedgehog.svg")
        self.place_in_area(hedgehog, "B3", "E6", scale_factor=1.5)
        
        cube_center = cube_3d.get_center()
        simulated_corners = list(sq_front.get_vertices()) + list(sq_back.get_vertices())
        
        spikes = VGroup()
        for corner in simulated_corners:
            direction = corner - cube_center
            # Extend them far out
            extended_end = cube_center + direction * 5.0 
            spikes.add(Line(cube_center, extended_end, color=COLOR_CORNER, stroke_width=1.0))

        self.play(
            FadeOut(cube_3d),
            FadeOut(sphere_speck),
            FadeOut(sphere_label),
            FadeOut(dist_label_3d),
            Create(spikes),
            FadeIn(hedgehog),
            run_time=3
        )
        
        # Highlight 'n' in formula
        n_part = formula_n[0][2] 
        self.play(n_part.animate.scale(1.5).set_color(YELLOW), run_time=0.5)
        self.play(n_part.animate.scale(1/1.5).set_color(COLOR_CORNER), run_time=0.5)

        self.wait(3)

        # Final Cleanup
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
