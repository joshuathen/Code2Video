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
        # 1. Setup the layout
        # Mandatory call with title and specific 3-line format.
        lecture_lines = [
            "Imagine any messy, loopy curve on a page.",
            "Can we find four points on its perimeter?",
            "Four points that form a perfect square?"
        ]
        self.setup_layout("The Hook: The Imperfect Doodle", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Imagine any messy, loopy curve on a page.
        # A messy, wobbly closed curve (coffee stain) is drawn in #C2B280.
        # The curve morphs into a different, even wigglier shape in #C2B280.
        self.play(self.lecture[0].animate.set_color("#C2B280"))
        
        def wobbly_path_1(t):
            angle = TAU * t
            r = 1.4 + 0.2 * np.sin(5 * angle) + 0.1 * np.cos(3 * angle)
            return np.array([r * np.cos(angle), r * np.sin(angle), 0])
            
        def wobbly_path_2(t):
            angle = TAU * t
            r = 1.4 + 0.3 * np.sin(7 * angle) + 0.15 * np.cos(4 * angle)
            return np.array([r * np.cos(angle), r * np.sin(angle), 0])

        curve = ParametricFunction(wobbly_path_1, t_range=[0, 1], color="#C2B280")
        curve_morphed = ParametricFunction(wobbly_path_2, t_range=[0, 1], color="#C2B280")
        
        # Resolve Issue 41: Reposition to B2-F6 and scale down to 0.8 to prevent crowding.
        self.place_in_area(curve, 'B2', 'F6', scale_factor=0.8)
        self.place_in_area(curve_morphed, 'B2', 'F6', scale_factor=0.8)
        
        self.play(Create(curve), run_time=2)
        self.play(ReplacementTransform(curve, curve_morphed), run_time=2)
        curve = curve_morphed
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Can we find four points on its perimeter?
        # Four white dots (#FFFFFF) appear at random positions on the curve.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFFFF")
        )
        
        # 4 random dots on the curve
        proportions_random = [0.05, 0.25, 0.5, 0.75]
        dots = VGroup(*[
            Dot(curve.point_from_proportion(p), color="#FFFFFF", radius=0.1)
            for p in proportions_random
        ])
        self.play(FadeIn(dots, shift=UP))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Four points that form a perfect square?
        # The dots slide along the curve and a faint yellow square (#FFFF00) connects them.
        # The square pulses in #FFFF00 and the label 'Square Peg Problem' appears in #FFFFFF.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        
        # Perfect square proportions (visual approximation on this specific curve)
        proportions_square = [0.12, 0.37, 0.62, 0.87]
        
        # Always redraw updates relative to dot movements
        square_edges = always_redraw(lambda: Polygon(
            *[d.get_center() for d in dots], 
            color="#FFFF00", 
            stroke_width=2,
            stroke_opacity=0.7
        ))
        self.add(square_edges)
        
        # Dots move along the curve to form a square
        self.play(
            dots[0].animate.move_to(curve.point_from_proportion(proportions_square[0])),
            dots[1].animate.move_to(curve.point_from_proportion(proportions_square[1])),
            dots[2].animate.move_to(curve.point_from_proportion(proportions_square[2])),
            dots[3].animate.move_to(curve.point_from_proportion(proportions_square[3])),
            run_time=2.5
        )
        
        # Label placement in the top center of the animation area using the grid
        label = Text("Square Peg Problem", font_size=24, color="#FFFFFF")
        self.place_at_grid(label, 'A4', scale_factor=1.0)
        
        # Pulse animation and Label appearance
        self.play(
            square_edges.animate.set_stroke(width=5, opacity=1),
            Write(label),
            run_time=1
        )
        self.play(square_edges.animate.set_stroke(width=2, opacity=0.7), run_time=1)
        
        self.wait(2)
