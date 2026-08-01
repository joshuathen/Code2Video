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
        # Setup the layout with specific lines for section 3
        lecture_lines = [
            'Complex roots of unity sit on the unit circle.',
            'Summing powers of omega creates a unique cancellation effect.',
            'If the index matches the root, they align perfectly.',
            'Otherwise, the vectors balance out to exactly zero.',
            "This 'zero or sum' property is our filter's engine."
        ]
        self.setup_layout("The Tool: Roots of Unity & The Circle of Cancellation", lecture_lines)

        # Colors
        COLOR_CIRCLE = "#555555"
        COLOR_ROOTS = "#00FFFF"
        COLOR_ALIGN = "#FF8800"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_ROOTS)
        
        # Complex Plane and Circle
        plane = NumberPlane(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=5,
            y_length=5,
            axis_config={"stroke_width": 2, "color": WHITE, "label_constructor": Text}
        ).add_coordinates()
        
        unit_circle = Circle(radius=plane.get_x_unit_size(), color=COLOR_CIRCLE, stroke_width=2)
        
        plane_group = VGroup(plane, unit_circle)
        # Resolved Issue 44: Expand to B1-F6 and set scale_factor=0.8
        self.place_in_area(plane_group, 'B1', 'F6', scale_factor=0.8)
        
        # Create vectors for n=3 roots of unity
        angles = [0, 2*PI/3, 4*PI/3]
        vectors = VGroup(*[
            Arrow(
                start=plane.coords_to_point(0, 0),
                end=plane.coords_to_point(np.cos(a), np.sin(a)),
                buff=0,
                color=COLOR_ROOTS,
                stroke_width=4
            ) for a in angles
        ])
        
        self.play(Create(plane), Create(unit_circle))
        self.play(Create(vectors))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_ROOTS)
        
        # Tip-to-tail animation
        v0, v1, v2 = vectors
        
        # Shift v1 to tip of v0
        # Shift v2 to tip of v1 (after v1 has moved)
        self.play(
            v1.animate.shift(v0.get_end() - v1.get_start()),
            run_time=1.5
        )
        self.play(
            v2.animate.shift(v1.get_end() - v2.get_start()),
            run_time=1.5
        )
        
        # Show they sum to zero (tip of v2 is back at origin)
        dot_origin = Dot(plane.coords_to_point(0,0), color=RED)
        self.play(Flash(dot_origin))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_ALIGN)
        
        # Reset vectors to origin and transform to 3rd powers (all point to 1)
        # All roots omega^0, omega^1, omega^2 raised to power 3 are 1.
        target_v = Arrow(
            start=plane.coords_to_point(0, 0),
            end=plane.coords_to_point(1, 0),
            buff=0,
            color=COLOR_ALIGN,
            stroke_width=4
        )
        
        self.play(
            vectors.animate.set_color(COLOR_ALIGN),
            v0.animate.move_to(plane.coords_to_point(0.5, 0)), # Center of arrow at 0.5
            v1.animate.move_to(plane.coords_to_point(0.5, 0)),
            v2.animate.move_to(plane.coords_to_point(0.5, 0)),
            run_time=2
        )
        # Re-syncing start/end explicitly for transformation visual
        self.play(
            ReplacementTransform(v0, target_v.copy()),
            ReplacementTransform(v1, target_v.copy()),
            ReplacementTransform(v2, target_v.copy()),
        )
        # Re-identify aligned_vectors
        aligned_vectors = VGroup(*[target_v.copy() for _ in range(3)])
        self.remove(v0, v1, v2)
        self.add(aligned_vectors)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_ROOTS)
        # Just a visual emphasis on the current alignment vs the previous balance
        self.play(Indicate(aligned_vectors))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        
        # Sum the three overlapping vectors to show length 3
        # We'll stretch them out tip to tail along the x-axis
        sum_v1 = aligned_vectors[1]
        sum_v2 = aligned_vectors[2]
        
        self.play(
            sum_v1.animate.shift(RIGHT * plane.get_x_unit_size()),
            sum_v2.animate.shift(RIGHT * 2 * plane.get_x_unit_size()),
            run_time=2
        )
        
        brace = Brace(VGroup(aligned_vectors[0], sum_v1, sum_v2), DOWN, color=WHITE)
        brace_label = Text("Sum = 3", font_size=24, color=WHITE).next_to(brace, DOWN)
        
        self.play(Create(brace), Write(brace_label))
        self.wait(2)
