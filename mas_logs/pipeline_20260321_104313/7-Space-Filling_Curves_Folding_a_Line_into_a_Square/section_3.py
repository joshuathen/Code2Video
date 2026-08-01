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
        # Setup layout with title and lecture lines
        lecture_lines = [
            "The Hilbert curve begins with a basic U-shape.",
            "We rotate and flip this shape in each quadrant.",
            "Connecting segments join these pieces into a single line.",
            "Increasing iterations fills the square with more detail.",
            "Eventually, the line becomes a solid block of color."
        ]
        self.setup_layout("The Hilbert Curve Construction", lecture_lines)

        # Helper function for Hilbert Curve coordinates (d to x,y)
        def d2xy(n, d):
            x = y = 0
            t = d
            s = 1
            while s < n:
                rx = 1 & (t // 2)
                ry = 1 & (t ^ rx)
                # Rotation logic
                if ry == 0:
                    if rx == 1:
                        x, y = s - 1 - y, s - 1 - x
                    else:
                        x, y = y, x
                x += s * rx
                y += s * ry
                t //= 4
                s *= 2
            return x, y

        def get_hilbert_coords(n, side_length=3.5):
            size = 2**n
            coords = []
            for i in range(size * size):
                ix, iy = d2xy(size, i)
                # Local centering: normalize to [-0.5, 0.5] then scale
                if size > 1:
                    px = (ix / (size - 1) - 0.5) * side_length
                    py = (iy / (size - 1) - 0.5) * side_length
                else:
                    px, py = 0, 0
                coords.append([px, py, 0])
            return coords

        # === Animation for Lecture Line 1 ===
        # Color: #00FFFF (Cyan)
        self.lecture[0].set_color("#00FFFF")
        h1_coords = get_hilbert_coords(1, side_length=2.5)
        h1_curve = VMobject(color="#00FFFF").set_points_as_corners(h1_coords)
        self.place_in_area(h1_curve, "A2", "F5")
        self.play(Create(h1_curve))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color: #FF66FF (Light Pink)
        self.lecture[1].set_color("#FF66FF")
        h2_coords = get_hilbert_coords(2, side_length=3.5)
        
        # Split into 4 quadrants (4 points each for n=2)
        q1 = VMobject(color="#FF66FF").set_points_as_corners(h2_coords[0:4])
        q2 = VMobject(color="#FF66FF").set_points_as_corners(h2_coords[4:8])
        q3 = VMobject(color="#FF66FF").set_points_as_corners(h2_coords[8:12])
        q4 = VMobject(color="#FF66FF").set_points_as_corners(h2_coords[12:16])
        
        quadrants = VGroup(q1, q2, q3, q4)
        self.place_in_area(quadrants, "A2", "F5") # Maintains relative internal positions
        
        self.play(
            ReplacementTransform(h1_curve, quadrants)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color: #ADD8E6 (Light Blue)
        self.lecture[2].set_color("#ADD8E6")
        h2_full = VMobject(color="#ADD8E6").set_points_as_corners(h2_coords)
        self.place_in_area(h2_full, "A2", "F5")
        
        self.play(
            ReplacementTransform(quadrants, h2_full)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Color: #FFFF00 (Yellow)
        self.lecture[3].set_color("#FFFF00")
        
        # Order 3
        h3_coords = get_hilbert_coords(3, side_length=3.5)
        h3_curve = VMobject(color="#FFFF00").set_points_as_corners(h3_coords)
        self.place_in_area(h3_curve, "A2", "F5")
        
        # Order 4
        h4_coords = get_hilbert_coords(4, side_length=3.5)
        h4_curve = VMobject(color="#FFFF00").set_points_as_corners(h4_coords)
        self.place_in_area(h4_curve, "A2", "F5")
        
        # Order 5
        h5_coords = get_hilbert_coords(5, side_length=3.5)
        h5_curve = VMobject(color="#FFFF00").set_points_as_corners(h5_coords)
        self.place_in_area(h5_curve, "A2", "F5")
        
        self.play(ReplacementTransform(h2_full, h3_curve))
        self.wait(0.5)
        self.play(ReplacementTransform(h3_curve, h4_curve))
        self.wait(0.5)
        self.play(ReplacementTransform(h4_curve, h5_curve))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Color: #FFFFFF (White)
        self.lecture[4].set_color("#FFFFFF")
        
        # Solid square block
        filled_square = Square(side_length=3.5, fill_opacity=1.0, fill_color=WHITE, stroke_width=0)
        self.place_in_area(filled_square, "A2", "F5")
        
        self.play(
            h5_curve.animate.set_color(WHITE),
            FadeIn(filled_square),
            run_time=2
        )
        self.wait(2)
