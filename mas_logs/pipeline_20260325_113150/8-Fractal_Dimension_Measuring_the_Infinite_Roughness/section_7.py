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

class Section7Scene(TeachingScene):
    def construct(self):
        # Initialize the layout
        self.setup_layout("Summary and Conclusion", [
            "Fractals bridge math and the natural world.",
            "Dimension measures how patterns change with scale.",
            "Geometry is not just smooth; it is rough."
        ])

        # === Animation for Lecture Line 1 ===
        # Fractals bridge math and the natural world.
        self.lecture[0].set_color("#00FFFF")
        
        # 1. Square
        square = Square(side_length=1, color="#00FFFF")
        self.place_at_grid(square, "B2", scale_factor=0.6) # Issue 63
        
        # 2. Sierpinski Triangle (Simple manual construction for stability)
        def get_sierpinski(order, size):
            tri = Triangle().scale(size)
            points = VGroup(tri)
            for _ in range(order):
                new_points = VGroup()
                for p in points:
                    s = p.copy().scale(0.5)
                    h = s.get_height()
                    w = s.get_width()
                    new_points.add(
                        s.copy().shift(UP * h/3),
                        s.copy().shift(LEFT * w/4 + DOWN * h/6),
                        s.copy().shift(RIGHT * w/4 + DOWN * h/6)
                    )
                points = new_points
            return points

        sierpinski = get_sierpinski(2, 0.6).set_color("#FF5555")
        self.place_at_grid(sierpinski, "B4", scale_factor=0.6) # Issue 63
        
        # 3. Koch Snowflake (Simplified visual representation)
        koch = Star(n=6, outer_radius=0.5, inner_radius=0.3, color="#00FF00")
        self.place_at_grid(koch, "B6", scale_factor=0.6) # Issue 63
        
        shapes = VGroup(square, sierpinski, koch)
        
        self.play(
            Create(shapes),
            run_time=2
        )
        self.play(
            Rotate(square, angle=PI/2),
            Rotate(sierpinski, angle=PI/2),
            Rotate(koch, angle=PI/2),
            run_time=2,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Dimension measures how patterns change with scale.
        self.lecture[1].set_color("#FFFFFF")
        
        summary_text = Text("Dimension = Complexity of Scale", font_size=24, color="#FFFFFF", weight=BOLD)
        self.place_in_area(summary_text, "C2", "C6", scale_factor=0.8) # Issue 62
        
        self.play(FadeIn(summary_text))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Geometry is not just smooth; it is rough.
        self.lecture[2].set_color("#00FFFF")
        
        final_title = Text("Fractal Dimension", font_size=42, color="#00FFFF", weight=BOLD)
        self.place_in_area(final_title, "E2", "F5", scale_factor=1.0) # Issue 61
        
        # Convergence animation using Transform to stay within motion constraints
        self.play(
            FadeOut(summary_text),
            Transform(shapes, final_title),
            run_time=2
        )
        # Replacing the transformed group with the final text object for clarity
        self.remove(shapes)
        self.add(final_title)
        self.wait(3)
        
        # Final transition to black
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(1)
