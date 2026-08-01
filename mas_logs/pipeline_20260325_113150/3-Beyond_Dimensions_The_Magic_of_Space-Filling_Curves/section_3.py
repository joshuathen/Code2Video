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
    def get_hilbert_path(self, order, size):
        """Generates a Hilbert curve of a given order and physical size."""
        n = 2**order
        cell_size = size / n
        
        def d2xy(n_val, d):
            """Maps distance along curve to (x, y) coordinates."""
            x = y = 0
            s = 1
            t = d
            while s < n_val:
                rx = 1 & (t // 2)
                ry = 1 & (t ^ rx)
                # Rotation logic for Hilbert curve
                if ry == 0:
                    if rx == 1:
                        x, y = s - 1 - x, s - 1 - y
                    x, y = y, x
                x += s * rx
                y += s * ry
                t //= 4
                s *= 2
            return x, y

        points = []
        for d in range(n * n):
            hx, hy = d2xy(n, d)
            # Center the points relative to origin
            px = (hx - (n - 1) / 2.0) * cell_size
            py = (hy - (n - 1) / 2.0) * cell_size
            points.append(np.array([px, py, 0]))
            
        curve = VMobject(color="#FFFF00")
        curve.set_points_as_corners(points)
        return curve

    def get_grid_lines(self, order, size):
        """Generates internal grid lines for the recursive subdivision."""
        n = 2**order
        cell_size = size / n
        lines = VGroup()
        for i in range(1, n):
            # Vertical subdivision lines
            x_pos = -size / 2 + i * cell_size
            v_line = Line([x_pos, -size / 2, 0], [x_pos, size / 2, 0], stroke_width=1, color=GRAY, stroke_opacity=0.4)
            # Horizontal subdivision lines
            y_pos = -size / 2 + i * cell_size
            h_line = Line([-size / 2, y_pos, 0], [size / 2, y_pos, 0], stroke_width=1, color=GRAY, stroke_opacity=0.4)
            lines.add(v_line, h_line)
        return lines

    def construct(self):
        # Initial layout setup
        self.setup_layout("The Finite Iteration: Building the Hilbert Curve", [
            "We start by dividing a square into four quadrants.",
            "A simple U-shape connects the center of each quadrant.",
            "Each quadrant is then subdivided into four smaller ones.",
            "Smaller U-shapes are rotated and linked together precisely.",
            "With every step, the line becomes more complexly packed."
        ])

        # Define the main working area for the animation (Right side)
        square_size = 4.8
        main_square = Square(side_length=square_size, color=WHITE, stroke_width=2)
        # Fix for Issue 31: reduce scale factor to 0.8 to avoid overflow into lecture area
        self.place_in_area(main_square, "A1", "F6", scale_factor=0.8)
        sq_center = main_square.get_center()

        # === Animation for Lecture Line 1 ===
        # Divide square into 2x2 grid. Draw yellow 3-segment 'U' curve (#FFFF00).
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        grid1 = self.get_grid_lines(1, square_size)
        self.place_in_area(grid1, "A1", "F6", scale_factor=0.8)
        curve1 = self.get_hilbert_path(1, square_size)
        self.place_in_area(curve1, "A1", "F6", scale_factor=0.8)
        
        self.play(Create(main_square), Create(grid1))
        self.play(Create(curve1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition to 4x4 grid and replace the curve with the Level 2 Hilbert curve (#FFFF00).
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFF00")
        )
        
        grid2 = self.get_grid_lines(2, square_size)
        self.place_in_area(grid2, "A1", "F6", scale_factor=0.8)
        curve2 = self.get_hilbert_path(2, square_size)
        self.place_in_area(curve2, "A1", "F6", scale_factor=0.8)
        
        self.play(
            ReplacementTransform(grid1, grid2),
            ReplacementTransform(curve1, curve2)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition to 8x8 grid and replace the curve with the Level 3 Hilbert curve (#FFFF00).
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        
        grid3 = self.get_grid_lines(3, square_size)
        self.place_in_area(grid3, "A1", "F6", scale_factor=0.8)
        curve3 = self.get_hilbert_path(3, square_size)
        self.place_in_area(curve3, "A1", "F6", scale_factor=0.8)
        
        self.play(
            ReplacementTransform(grid2, grid3),
            ReplacementTransform(curve2, curve3)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Zoom in on a small section of Level 3 curve to highlight its recursive nature.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#FFFF00")
        )
        
        zoom_group = VGroup(main_square, grid3, curve3)
        # Target point for zoom: Center of the bottom-left quadrant.
        # Adjusted for the 0.8 scale factor applied via place_in_area.
        actual_size = square_size * 0.8
        zoom_target = sq_center + np.array([-actual_size / 4, -actual_size / 4, 0])
        
        self.play(
            zoom_group.animate.scale(2.5, about_point=zoom_target)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Show Level 5 Hilbert curve (#FFFF00) without grid lines.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color("#FFFF00")
        )
        
        # Reset visual state to a full square view showing Level 5 curve
        # Fix for Issue 32: set scale factor to 0.9 to prevent overlap with text
        curve5 = self.get_hilbert_path(5, square_size)
        self.place_in_area(curve5, "A1", "F6", scale_factor=0.9)
        final_square = Square(side_length=square_size, color=WHITE, stroke_width=2)
        self.place_in_area(final_square, "A1", "F6", scale_factor=0.9)
        
        self.play(
            FadeOut(zoom_group),
            FadeIn(final_square),
            Create(curve5)
        )
        self.wait(2)
