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
        # Initial Setup
        lecture_lines = [
            "Otto Toeplitz conjectured this back in 1911.",
            "It is proven for simple shapes like circles.",
            "But jagged, mysterious curves remain a challenge."
        ]
        self.setup_layout("Defining the Goal: Toeplitz's Conjecture", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Otto Toeplitz conjectured this back in 1911.
        self.lecture[0].set_color(YELLOW)
        
        # Text "Toeplitz's Conjecture" in #FFFFFF appears
        conjecture_label = Text("Toeplitz's Conjecture", font_size=32, color=WHITE)
        # Fix Issue 44: Moved area to B2-B4 and used scale_factor=0.8
        self.place_in_area(conjecture_label, "B2", "B4", scale_factor=0.8)
        
        # A pulsating blob in #FF00FF
        def get_blob_points(amplitude=0.2, phase=0):
            points = []
            for angle in np.linspace(0, 2 * PI, 100):
                r = 1.2 + amplitude * np.sin(3 * angle + phase)
                points.append([r * np.cos(angle), r * np.sin(angle), 0])
            return points

        blob = VMobject()
        blob.set_points_as_corners(get_blob_points())
        blob.set_color("#FF00FF")
        blob.set_stroke(width=4)
        # Fix Issue 45: Moved area to D3-E5
        self.place_in_area(blob, "D3", "E5", scale_factor=0.8)

        self.play(
            Write(conjecture_label),
            Create(blob),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # It is proven for simple shapes like circles.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Morph blob into a smooth circle
        circle_shape = Circle(radius=1.2, color="#FF00FF").set_stroke(width=4)
        # Fix Issue 45: Moved area to D3-E5
        self.place_in_area(circle_shape, "D3", "E5", scale_factor=0.8)
        
        # A yellow square (#FFFF00) inside
        square_side = 1.2 * np.sqrt(2)
        square = Square(side_length=square_side, color="#FFFF00")
        # Fix Issue 45: Moved area to D3-E5
        self.place_in_area(square, "D3", "E5", scale_factor=0.8)

        self.play(Transform(blob, circle_shape))
        self.play(Create(square))
        
        # Rotate square to show it maintains corners on the circle
        self.play(Rotate(square, angle=PI/2), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # But jagged, mysterious curves remain a challenge.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Change blob to a "jagged" shape
        def get_jagged_points(phase=0):
            points = []
            for angle in np.linspace(0, 2 * PI, 200):
                r = 1.2 + 0.1 * np.sin(5 * angle + phase) + 0.05 * np.cos(13 * angle + phase*1.5)
                points.append([r * np.cos(angle), r * np.sin(angle), 0])
            return points

        jagged_blob_base = VMobject().set_points_as_corners(get_jagged_points())
        jagged_blob_base.set_color("#FF00FF").set_stroke(width=4)
        # Fix Issue 46: Moved area to D3-E5
        self.place_in_area(jagged_blob_base, "D3", "E5", scale_factor=0.8)

        # Update the square to "search" erratically
        time_tracker = ValueTracker(0)
        self.add(time_tracker)
        time_tracker.add_updater(lambda m, dt: m.increment_value(dt))

        def searching_square_updater(m):
            t = time_tracker.get_value()
            scale_val = 1.0 + 0.2 * np.sin(4 * t)
            new_sq = Square(side_length=square_side * scale_val, color="#FFFF00")
            new_sq.rotate(t * 1.5)
            # Fix Issue 46: Moved area to D3-E5
            self.place_in_area(new_sq, "D3", "E5", scale_factor=0.8)
            m.become(new_sq)

        def pulsating_blob_updater(m):
            t = time_tracker.get_value()
            new_pts = get_jagged_points(phase=t*2)
            new_m = VMobject().set_points_as_corners(new_pts)
            new_m.set_color("#FF00FF").set_stroke(width=4)
            # Fix Issue 46: Moved area to D3-E5
            self.place_in_area(new_m, "D3", "E5", scale_factor=0.8)
            m.become(new_m)

        self.play(Transform(blob, jagged_blob_base))
        
        square.add_updater(searching_square_updater)
        blob.add_updater(pulsating_blob_updater)
        
        self.wait(4)
        
        # Cleanup
        square.remove_updater(searching_square_updater)
        blob.remove_updater(pulsating_blob_updater)
        self.wait(1)
