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
        # Setup layout with title and lecture lines
        lecture_lines = [
            "At infinity, the 1D line occupies the 2D area.",
            "It challenges our basic intuition of geometry and dimension.",
            "A finite space holds an infinite, perfectly continuous journey."
        ]
        self.setup_layout("Summary and the Infinite Limit", lecture_lines)

        # Helper function for Hilbert Curve points (iterative d2xy)
        def get_hilbert_points(order, scale_val):
            n = 2**order
            points = []
            for i in range(n * n):
                x, y = 0, 0
                t = i
                s = 1
                while s < n:
                    rx = 1 & (t // 2)
                    ry = 1 & (t ^ rx)
                    # rot
                    if ry == 0:
                        if rx == 1:
                            x, y = s - 1 - x, s - 1 - y
                        x, y = y, x
                    x += s * rx
                    y += s * ry
                    t //= 4
                    s *= 2
                # Center and scale
                points.append(np.array([
                    (x / (n - 1) - 0.5) * scale_val,
                    (y / (n - 1) - 0.5) * scale_val,
                    0
                ]))
            return points

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1 in Cyan
        self.play(self.lecture[0].animate.set_color("#00FFFF"))

        # Create Hilbert Curve (Order 5 for density)
        # B2 is (1.5, 1.2), E5 is (4.5, -1.8). Center (3.0, -0.3). Side length 3.0.
        order_5_points = get_hilbert_points(5, 3.0)
        hilbert_curve = VMobject(color="#00FFFF")
        hilbert_curve.set_points_as_corners(order_5_points)
        hilbert_curve.set_stroke(width=2)
        
        self.place_in_area(hilbert_curve, 'B2', 'E5')
        self.play(Create(hilbert_curve), run_time=3)

        # Zoom into the curve until it becomes a solid square
        # We simulate the "limit" by transforming the dense curve into a solid square
        limit_square = Square(side_length=3.0, fill_opacity=1.0, fill_color="#00FFFF", stroke_width=0)
        self.place_in_area(limit_square, 'B2', 'E5')

        self.play(
            hilbert_curve.animate.scale(1.5),
            ReplacementTransform(hilbert_curve, limit_square),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2 in White
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))

        # Display text '1D Path fills 2D Space' in #FFFFFF at the center
        limit_text = Text("1D Path fills 2D Space", font_size=24, color="#FFFFFF")
        # Place text slightly above center to not be fully obscured if needed, 
        # but prompt says "at the center".
        self.place_in_area(limit_text, 'C3', 'D4')
        
        self.play(Write(limit_text))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3 in Green
        self.play(self.lecture[2].animate.set_color("#00FF00"))

        # The filled square pulses with a soft #00FF00 glow before fading to black.
        pulse_overlay = limit_square.copy().set_fill("#00FF00", opacity=0.0)
        
        self.play(
            pulse_overlay.animate.set_fill(opacity=0.6).scale(1.05),
            limit_square.animate.set_color("#00FF00"),
            run_time=1,
            rate_func=there_and_back
        )
        self.wait(1)

        # Fade everything to black
        self.play(
            FadeOut(limit_square),
            FadeOut(limit_text),
            FadeOut(pulse_overlay),
            FadeOut(self.lecture),
            FadeOut(self.title),
            run_time=2
        )
        self.wait(1)
