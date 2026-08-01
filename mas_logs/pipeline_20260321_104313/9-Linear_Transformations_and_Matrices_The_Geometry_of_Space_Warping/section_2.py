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
        # Setup the layout with title and lecture lines as requested
        title = "What is a 'Linear' Transformation?"
        lecture_lines = [
            "A transformation moves every point on our coordinate grid.",
            "Linear transforms rotate and scale the space evenly.",
            "Notice that grid lines always remain parallel and straight.",
            "Squiggly or curved warping is not a linear transformation.",
            "Critically, the origin must never move from the center."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors as per instructions and animation description
        GRID_COLOR = "#444444"
        DOT_COLOR = "#FFFFFF"
        HIGHLIGHT_COLOR = "#00FFFF"
        TEXT_HIGHLIGHT = YELLOW
        VECTOR_COLOR = "#FF00FF"

        # Create Grid visuals
        # Grid covers a wider range to ensure no empty space during rotation/scaling
        grid_lines = VGroup()
        for i in range(-6, 7):
            grid_lines.add(Line(np.array([i, -6, 0]), np.array([i, 6, 0]), color=GRID_COLOR, stroke_width=1.5))
            grid_lines.add(Line(np.array([-6, i, 0]), np.array([6, i, 0]), color=GRID_COLOR, stroke_width=1.5))
        
        origin_dot = Dot(point=ORIGIN, color=DOT_COLOR, radius=0.08)
        
        # Add a few vectors to demonstrate mapping
        v1 = Arrow(ORIGIN, [1.2, 0.8, 0], buff=0, color=VECTOR_COLOR, stroke_width=4)
        v2 = Arrow(ORIGIN, [-0.8, 1.5, 0], buff=0, color=VECTOR_COLOR, stroke_width=4)
        vectors = VGroup(v1, v2)

        grid_visuals = VGroup(grid_lines, origin_dot, vectors)
        
        # Resolve Issue 36: Optimize grid area (B2 to F5) and scale factor (0.9)
        self.place_in_area(grid_visuals, 'B2', 'F5', scale_factor=0.9)
        
        # Visual anchor for transformations
        center_pos = origin_dot.get_center().copy()

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(TEXT_HIGHLIGHT))
        self.play(Create(grid_lines), FadeIn(origin_dot), Create(vectors))
        self.wait(0.5)
        # Demonstrate mapping by shifting the entire space slightly
        self.play(
            grid_visuals.animate.shift(RIGHT * 0.4 + UP * 0.2),
            run_time=1.5
        )
        self.wait(0.5)
        # Return to center before defining linear properties
        self.play(
            grid_visuals.animate.move_to(center_pos),
            run_time=1
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(TEXT_HIGHLIGHT)
        )
        # Linear transform: rotate and scale
        self.play(
            Rotate(grid_lines, angle=35 * DEGREES, about_point=center_pos),
            grid_lines.animate.scale(0.8, about_point=center_pos),
            Rotate(vectors, angle=35 * DEGREES, about_point=center_pos),
            vectors.animate.scale(0.8, about_point=center_pos),
            Indicate(origin_dot, color=DOT_COLOR, scale_factor=1.3),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(TEXT_HIGHLIGHT)
        )
        # Highlight parallel lines (changing grid color to cyan)
        self.play(
            grid_lines.animate.set_color(HIGHLIGHT_COLOR),
            grid_lines.animate.scale(1.2, about_point=center_pos),
            vectors.animate.scale(1.2, about_point=center_pos),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(TEXT_HIGHLIGHT)
        )
        
        grid_lines.save_state()
        vectors.save_state()
        
        # Define a non-linear (squiggly) distortion
        def wavy_distortion(p):
            rel_p = p - center_pos
            dx = 0.35 * np.sin(rel_p[1] * 2.5)
            dy = 0.35 * np.cos(rel_p[0] * 2.5)
            return center_pos + rel_p + np.array([dx, dy, 0])

        # Apply squiggly warping to demonstrate non-linearity
        self.play(
            grid_lines.animate.apply_function(wavy_distortion),
            vectors.animate.apply_function(wavy_distortion),
            run_time=2.5
        )
        self.wait(1)
        # Restore grid to linear state
        self.play(Restore(grid_lines), Restore(vectors), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(TEXT_HIGHLIGHT)
        )
        # Emphasize that the origin is the fixed point of the transformation
        self.play(Indicate(origin_dot, color=DOT_COLOR, scale_factor=2.5))
        self.wait(2)
        
        # Cleanup
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
