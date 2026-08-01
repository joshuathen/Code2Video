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
        # Setup layout with lecture lines
        lecture_lines = [
            "Euler's formula describes every point on this circle.",
            "The variable x represents the distance traveled.",
            "Cosine and sine track the horizontal and vertical positions."
        ]
        self.setup_layout("Connecting the Dots: Euler's Formula Visualized", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Euler's formula in white - positioned at top of grid area (A1-A6)
        euler_formula = Text("e^(ix) = cos(x) + i sin(x)", color=WHITE)
        self.place_in_area(euler_formula, "A1", "A6", scale_factor=0.6)
        
        # Coordinate Plane - positioned in the main grid area (B1-E6)
        # Simplified NumberPlane (no coordinates) to ensure rendering fits within time limit
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            axis_config={"stroke_opacity": 0.5},
            background_line_style={"stroke_opacity": 0.2}
        )
        self.place_in_area(plane, "B1", "E6", scale_factor=0.8)
        
        # Unit Circle
        unit_circle = Circle(radius=plane.get_x_unit_size(), color=WHITE, stroke_opacity=0.5)
        unit_circle.move_to(plane.coords_to_point(0,0))
        
        self.play(Write(euler_formula), run_time=1)
        self.play(Create(plane), Create(unit_circle), run_time=1)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Variable x represents distance/angle traveled
        self.play(self.lecture[1].animate.set_color("#00FF00"), run_time=0.5)
        
        x_tracker = ValueTracker(0)
        
        # Vector in green (#00FF00)
        vector = Line(
            start=plane.coords_to_point(0,0),
            end=plane.coords_to_point(1,0),
            color="#00FF00",
            stroke_width=6
        )
        def update_vector(v):
            val = x_tracker.get_value()
            v.put_start_and_end_on(
                plane.coords_to_point(0,0),
                plane.coords_to_point(np.cos(val), np.sin(val))
            )
        vector.add_updater(update_vector)
        
        # Label for angle x
        x_label = Text("x", color="#00FF00", font_size=24)
        def update_x_label(l):
            val = x_tracker.get_value()
            l.move_to(plane.coords_to_point(
                1.2 * np.cos(val), 
                1.2 * np.sin(val)
            ))
        x_label.add_updater(update_x_label)
        
        self.play(Create(vector), Write(x_label), run_time=1)
        self.play(x_tracker.animate.set_value(PI/3), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Cosine (red, #FF0000) and Sine (blue, #0000FF) projections
        self.play(self.lecture[2].animate.set_color("#00FFFF"), run_time=0.5) # Light blue mix for highlight
        
        cos_line = Line(color="#FF0000", stroke_width=8)
        def update_cos_line(cl):
            val = x_tracker.get_value()
            cl.put_start_and_end_on(
                plane.coords_to_point(0,0),
                plane.coords_to_point(np.cos(val), 0)
            )
        cos_line.add_updater(update_cos_line)
        
        sin_line = Line(color="#0000FF", stroke_width=8)
        def update_sin_line(sl):
            val = x_tracker.get_value()
            sl.put_start_and_end_on(
                plane.coords_to_point(np.cos(val), 0),
                plane.coords_to_point(np.cos(val), np.sin(val))
            )
        sin_line.add_updater(update_sin_line)
        
        self.play(Create(cos_line), Create(sin_line), run_time=1)
        
        # Move to PI to show transition to Euler's Identity
        self.play(x_tracker.animate.set_value(PI), run_time=2)
        
        # Euler's Identity result at bottom of grid area (F1-F6)
        euler_identity = Text("e^(i*PI) = -1", color=GOLD)
        self.place_in_area(euler_identity, "F1", "F6", scale_factor=0.7)
        
        self.play(Write(euler_identity), run_time=1)
        self.play(Indicate(euler_identity), run_time=1)
        self.wait(1)

if __name__ == "__main__":
    pass
