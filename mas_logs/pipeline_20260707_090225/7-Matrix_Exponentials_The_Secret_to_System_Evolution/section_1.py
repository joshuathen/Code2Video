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
        # Setup layout
        self.setup_layout(
            "The Growth Analogy: From Scalars to Matrices", 
            [
                'Simple growth follows the equation dx/dt equals rx.', 
                'The solution is the familiar exponential function, e to rt.', 
                'In systems, matrices replace scalars to describe multi-variable growth.'
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Initial state: Line 1 highlighted
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # Replacing MathTex with Text to avoid FileNotFoundError: 'latex'
        eq1 = Text("dx/dt = rx", font_size=32, color="#FFFFFF")
        # Start in the middle of the visual area
        # Fix for Issue 32: Adjusted placement and scale
        self.place_in_area(eq1, 'A2', 'B3', scale_factor=1.0)
        self.play(Write(eq1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition highlight
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFFFF")
        )
        
        # Solution equation
        eq2 = Text("x(t) = x(0)e^(rt)", font_size=32, color="#FFFFFF")
        
        # Growth circle
        circle = Circle(radius=0.5, color="#00FFFF", stroke_width=4)
        # Position circle in the lower-left grid area
        self.place_in_area(circle, "D2", "E2", scale_factor=1.0)
        
        # Move eq1 and transform it into eq2
        # Fix for Issue 32: Adjusted target position
        self.play(
            Transform(eq1, self.place_at_grid(eq2, 'A2', scale_factor=0.8)),
            Create(circle)
        )
        
        # Animate scalar growth (Circle scaling)
        self.play(circle.animate.scale(2), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight matrix concept line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FFFF")
        )
        
        # Matrix equation on the right side of the grid
        # Fix for Issue 33: Adjusted placement to avoid plane grid lines
        eq3 = Text("dx/dt = Ax", font_size=32, color="#00FFFF")
        self.place_at_grid(eq3, 'A5', scale_factor=0.9)
        
        # Coordinate system on the bottom right
        # Fix for Issue 34: Adjusted placement to avoid circle overlap
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=3,
            y_length=3,
            background_line_style={"stroke_opacity": 0.4}
        )
        self.place_in_area(plane, 'E4', 'F6', scale_factor=1.0)
        
        # Basis vectors
        v1 = Vector([1, 0], color="#FF0000")
        v2 = Vector([0, 1], color="#00FF00")
        plane.add(v1, v2)
        
        # Dots for tracing paths
        dot1 = Dot(v1.get_end(), radius=0).set_opacity(0)
        dot2 = Dot(v2.get_end(), radius=0).set_opacity(0)
        plane.add(dot1, dot2)
        
        # Traced paths
        path1 = TracedPath(dot1.get_center, stroke_color="#FF0000", stroke_width=3)
        path2 = TracedPath(dot2.get_center, stroke_color="#00FF00", stroke_width=3)
        
        self.play(
            Write(eq3),
            Create(plane)
        )
        self.add(path1, path2)
        
        # Matrix transformation over time
        t_final = 1.2
        val = 0.2 * t_final
        rot = 0.7 * t_final
        # Final transformation matrix
        mat_final = np.exp(val) * np.array([
            [np.cos(rot), -np.sin(rot)],
            [np.sin(rot), np.cos(rot)]
        ])
        
        # Animate the linear transformation over time
        self.play(
            ApplyMatrix(mat_final, plane),
            run_time=4,
            rate_func=linear
        )
        self.wait(2)
