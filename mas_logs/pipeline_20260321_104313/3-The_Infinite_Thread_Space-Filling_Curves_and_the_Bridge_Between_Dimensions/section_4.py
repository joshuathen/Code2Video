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
        # Title and Lecture Lines
        title_text = "The Finite vs. The Infinite Limit"
        lecture_lines = [
            "For any finite step, the curve has zero area.",
            "At the infinite limit, its length becomes truly infinite.",
            "Remarkably, the curve then visits every point in the square."
        ]
        self.setup_layout(title_text, lecture_lines)

        def get_hilbert_curve(order):
            """Generates a Hilbert curve of a given order normalized to a [0, 1] x [0, 1] box."""
            n = 2**order
            pts = []
            for d in range(n * n):
                # Distance d to (x,y) coordinates
                t, x, y, s = d, 0, 0, 1
                while s < n:
                    rx = 1 & (t // 2)
                    ry = 1 & (t ^ rx)
                    # Coordinate rotation logic
                    if ry == 0:
                        if rx == 1:
                            x, y = s - 1 - x, s - 1 - y
                        x, y = y, x
                    x += s * rx
                    y += s * ry
                    t //= 4
                    s *= 2
                # Normalize points to [0, 1]
                divisor = (n - 1) if n > 1 else 1
                pts.append([x / divisor, y / divisor, 0])
            
            curve = VMobject().set_points_as_corners(pts)
            curve.set_color(WHITE)
            curve.set_stroke(width=2)
            return curve

        # === Animation for Lecture Line 1 ===
        # Highlight line 1 in Red to match the 'Area' label
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        
        # Show a 4th-order Hilbert curve in #FFFFFF
        curve4 = get_hilbert_curve(4)
        self.place_in_area(curve4, 'B2', 'E5', scale_factor=3.0)
        
        # Label 'Area = 0' in #FF0000
        label_area = Text("Area = 0", font_size=24, color="#FF0000")
        self.place_in_area(label_area, 'A3', 'A4', scale_factor=1.0)
        
        self.play(Create(curve4), Write(label_area), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Revert Line 1 and Highlight Line 2 in Cyan
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FFFF")
        )
        
        # Higher order curve to simulate the "multiplication" of segments
        curve6 = get_hilbert_curve(6)
        self.place_in_area(curve6, 'B2', 'E5', scale_factor=3.0)
        
        # Label 'Length = ∞' in #00FFFF
        label_length = Text("Length = ∞", font_size=24, color="#00FFFF")
        self.place_in_area(label_length, 'F3', 'F4', scale_factor=1.0)
        
        self.play(
            Transform(curve4, curve6),
            Write(label_length),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Revert Line 2 and Highlight Line 3 in White
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(WHITE)
        )
        
        # Solid white square to represent the curve visiting every point
        final_square = Square(side_length=1.0, fill_opacity=1, fill_color=WHITE, stroke_width=0)
        self.place_in_area(final_square, 'B2', 'E5', scale_factor=3.0)
        
        # The white lines thicken and merge
        self.play(curve4.animate.set_stroke(width=15), run_time=2)
        self.play(FadeIn(final_square))
        self.wait(2)
