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
        # Initial layout setup
        self.setup_layout("Conclusion: Dimension is a Matter of Perspective", [
            "Infinite iteration transforms a simple rule into dense reality.",
            "The boundary between dimensions begins to blur and fade.",
            "A single thread becomes a solid, space-occupying square."
        ])

        # Helper function for Hilbert Curve points
        def get_hilbert_points(order, scale_factor):
            def rot(n, x, y, rx, ry):
                if ry == 0:
                    if rx == 1:
                        x = n - 1 - x
                        y = n - 1 - y
                    return y, x
                return x, y

            n = 2**order
            pts = []
            for i in range(n*n):
                tx, ty = 0, 0
                t = i
                s = 1
                while s < n:
                    rx = 1 & (t // 2)
                    ry = 1 & (t ^ rx)
                    tx, ty = rot(s, tx, ty, rx, ry)
                    tx += s * rx
                    ty += s * ry
                    t //= 4
                    s *= 2
                pts.append(np.array([tx, ty, 0]))
            
            # Center and scale points
            pts = np.array(pts, dtype=float)
            pts -= np.mean(pts, axis=0)
            # Normalize to fit roughly within a unit square before place_in_area handles final scaling
            max_range = np.max(pts) - np.min(pts)
            if max_range > 0:
                pts /= max_range
            return pts

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Create a high-order Hilbert curve (order 5 provides a dense mesh look)
        h_pts = get_hilbert_points(order=5, scale_factor=4.0)
        hilbert_curve = VMobject(color=WHITE, stroke_width=1.5)
        hilbert_curve.set_points_as_corners(h_pts)
        
        # Position in the designated area
        self.place_in_area(hilbert_curve, "A1", "F6", scale_factor=4.5)
        
        self.play(Create(hilbert_curve), run_time=5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Blurring effect by increasing stroke width and adding a slight glow/expansion
        self.play(
            hilbert_curve.animate.set_stroke(width=8),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Create the solid square
        solid_square = Square(side_length=1.0, fill_opacity=1.0, fill_color=WHITE, stroke_width=0)
        # Position it in the same area as the curve
        self.place_in_area(solid_square, "A1", "F6", scale_factor=4.5)
        
        # Final transition to solid square
        self.play(
            ReplacementTransform(hilbert_curve, solid_square),
            run_time=2
        )
        self.wait(3)
        
        # Reset last line color
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
