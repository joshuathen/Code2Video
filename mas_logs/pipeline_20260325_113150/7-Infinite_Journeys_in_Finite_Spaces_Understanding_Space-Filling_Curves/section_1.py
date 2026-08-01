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
        # Initial layout setup
        title = "The Geometric Paradox"
        lines = [
            "A line has one dimension, while a square has two.",
            "Intuitively, a line cannot cover every point in space.",
            "Imagine a path that visits every coordinate perfectly.",
            "This mathematical marvel is called a space-filling curve.",
            "Can a continuous line truly fill an entire area?"
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # 1D Line
        line_1d = Line(LEFT, RIGHT, color="#00FF00", stroke_width=4)
        self.place_at_grid(line_1d, "B2", scale_factor=0.8)
        line_label = Text("1D Line", font_size=18, color="#00FF00").next_to(line_1d, UP, buff=0.1)
        
        # 2D Square
        square_2d = Square(side_length=2, color="#0080FF", fill_opacity=0.4, stroke_width=2)
        self.place_at_grid(square_2d, "B5", scale_factor=0.8)
        square_label = Text("2D Square", font_size=18, color="#0080FF").next_to(square_2d, UP, buff=0.1)

        self.play(Create(line_1d), Create(line_label))
        self.play(Create(square_2d), Create(square_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Clear specific elements to focus on the square
        self.play(FadeOut(line_1d), FadeOut(line_label), FadeOut(square_label))
        # Move square to a larger central area for the next animations
        self.play(self.place_in_area(square_2d, "C2", "F5", scale_factor=1.2).animate)

        # Random dot movement inside the square
        dot = Dot(color=WHITE, radius=0.05)
        dot.move_to(square_2d.get_center())
        
        random_path = VMobject(color=WHITE, stroke_width=1)
        random_path.set_points_as_corners([dot.get_center()])

        def update_path(path):
            new_point = dot.get_center()
            path.add_points_as_corners([new_point])

        self.add(dot, random_path)
        
        # Simulate "random" movement within boundaries
        s_side = square_2d.side_length * 0.9 / 2
        c = square_2d.get_center()
        
        points = [
            c + np.array([np.random.uniform(-s_side, s_side), np.random.uniform(-s_side, s_side), 0])
            for _ in range(8)
        ]
        
        random_path.add_updater(update_path)
        for p in points:
            self.play(dot.animate.move_to(p), run_time=0.4, rate_func=linear)
        
        random_path.remove_updater(update_path)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        self.play(FadeOut(dot), FadeOut(random_path))

        # Helper to generate Hilbert Curve points
        def get_hilbert_points(n, side=2.0):
            def rot(n, x, y, rx, ry):
                if ry == 0:
                    if rx == 1:
                        x = n - 1 - x
                        y = n - 1 - y
                    return y, x
                return x, y

            points = []
            for i in range(4**n):
                t = i
                x = 0
                y = 0
                s = 1
                while s < 2**n:
                    rx = 1 & (t // 2)
                    ry = 1 & (t ^ rx)
                    x, y = rot(s, x, y, rx, ry)
                    x += s * rx
                    y += s * ry
                    t //= 4
                    s *= 2
                points.append(np.array([x, y, 0]))
            
            # Normalize and center
            points = np.array(points)
            points = points - (2**n - 1) / 2
            points = points * (side / (2**n))
            return points

        # Create the dense path (Hilbert Curve Order 4)
        h_points = get_hilbert_points(4, side=square_2d.side_length * 1.5)
        h_points += square_2d.get_center() # Align with square
        
        hilbert_curve = VMobject(color="#FFFF00", stroke_width=1.5)
        hilbert_curve.set_points_as_corners(h_points)

        self.play(Create(hilbert_curve), run_time=5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        curve_label = Text("Space-Filling Curve", font_size=24, color="#FFFF00")
        self.place_at_grid(curve_label, "B3", scale_factor=1.0) # Positioned above the square area
        
        self.play(Write(curve_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        self.play(FadeOut(curve_label))
        
        # Increase thickness until it fills the square
        self.play(
            hilbert_curve.animate.set_stroke(width=15),
            run_time=3
        )
        self.wait(2)
