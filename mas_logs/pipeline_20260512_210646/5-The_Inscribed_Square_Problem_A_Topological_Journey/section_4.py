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
        lecture_lines = [
            'We map pairs of points into a 3D space.',
            'Unordered pairs on a circle form a Möbius strip.',
            "The curve's points map to the strip's boundary edge.",
            'This topological transformation reveals hidden geometric connections.',
            'Watch how the loop twists into a 3D surface.'
        ]
        self.setup_layout("The Topological Trick: The Möbius Strip Mapping", lecture_lines)
        
        # Define shared colors
        COLOR_SQUARE = "#808080"
        COLOR_MOB_SURFACE = "#DA70D6"
        COLOR_DIAGONAL = "#FFFFFF"
        COLOR_TRANSFORM = "#ADD8E6"
        COLOR_TWIST = "#F0E68C"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_SQUARE)
        
        # Define the square representing point pairs (A, B)
        square = Rectangle(
            height=3.8, width=3.8, 
            fill_opacity=0.3, 
            fill_color=COLOR_SQUARE, 
            stroke_color=COLOR_SQUARE
        )
        self.place_in_area(square, 'B2', 'E5')
        
        label_a = Text("Point A position", font_size=16, color=COLOR_SQUARE)
        label_b = Text("Point B position", font_size=16, color=COLOR_SQUARE).rotate(PI/2)
        
        # Positioning labels using the grid area system
        self.place_in_area(label_a, 'F3', 'F4', scale_factor=0.8)
        self.place_in_area(label_b, 'B1', 'E1', scale_factor=0.8)

        self.play(FadeIn(square), Write(label_a), Write(label_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_MOB_SURFACE)
        
        arrow_left = Arrow(
            start=square.get_corner(DL), end=square.get_corner(UL),
            color=COLOR_MOB_SURFACE, buff=0, stroke_width=5
        )
        arrow_right = Arrow(
            start=square.get_corner(UR), end=square.get_corner(DR),
            color=COLOR_MOB_SURFACE, buff=0, stroke_width=5
        )
        
        glue_note = Text("Identification with a Twist", font_size=18, color=COLOR_MOB_SURFACE)
        # Fix: Better alignment for glue note
        self.place_in_area(glue_note, 'A3', 'A4', scale_factor=0.8)
        
        self.play(Create(arrow_left), Create(arrow_right), Write(glue_note))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_DIAGONAL)
        
        diagonal = Line(square.get_corner(DL), square.get_corner(UR), color=COLOR_DIAGONAL, stroke_width=6)
        diag_label = Text("Diagonal (A = B)", font_size=16, color=COLOR_DIAGONAL)
        diag_label.rotate(45 * DEGREES)
        # Position label on the grid near the diagonal
        self.place_at_grid(diag_label, 'C3', scale_factor=0.8)
        
        self.play(Create(diagonal), Write(diag_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_TRANSFORM)
        
        def mobius_strip_boundary(t):
            return np.array([
                (2.0 + 0.6 * np.cos(t/2)) * np.cos(t),
                (2.0 + 0.6 * np.cos(t/2)) * np.sin(t) * 0.5,
                0
            ])

        mob_surface_rep = ParametricFunction(
            mobius_strip_boundary, t_range=[0, 2*TAU], 
            color=COLOR_MOB_SURFACE
        ).set_stroke(width=25, opacity=0.6)
        
        mob_edge_rep = mob_surface_rep.copy().set_color(COLOR_DIAGONAL).set_stroke(width=4, opacity=1.0)
        
        # Fix: Increased scale factor for visibility
        self.place_in_area(mob_surface_rep, 'B2', 'E5', scale_factor=0.8)
        self.place_in_area(mob_edge_rep, 'B2', 'E5', scale_factor=0.8)

        self.play(
            ReplacementTransform(square, mob_surface_rep),
            ReplacementTransform(diagonal, mob_edge_rep),
            FadeOut(label_a), FadeOut(label_b), 
            FadeOut(glue_note), FadeOut(diag_label),
            FadeOut(arrow_left), FadeOut(arrow_right),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(COLOR_TWIST)
        
        boundary_info = Text("Single Boundary Edge", font_size=20, color=COLOR_DIAGONAL)
        self.place_at_grid(boundary_info, 'F4')
        
        self.play(
            mob_edge_rep.animate.set_stroke(width=8),
            Write(boundary_info),
            run_time=1
        )
        
        self.play(
            mob_surface_rep.animate.rotate(20*DEGREES, axis=RIGHT),
            mob_edge_rep.animate.rotate(20*DEGREES, axis=RIGHT),
            run_time=1.5
        )
        self.play(
            mob_surface_rep.animate.rotate(-40*DEGREES, axis=RIGHT),
            mob_edge_rep.animate.rotate(-40*DEGREES, axis=RIGHT),
            run_time=1.5
        )
        
        self.wait(2)
