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
        # Setup layout with lecture lines as specified
        # Fixed: Removed LaTeX math delimiters ($x$) to avoid LaTeX rendering error
        lecture_lines = [
            "Euler's formula describes every point on this circle.",
            "The variable x represents the distance traveled.",
            "Cosine and sine track the horizontal and vertical positions."
        ]
        self.setup_layout("Connecting the Dots: Euler's Formula Visualized", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Fixed: Switched from MathTex to Text to bypass missing 'latex' executable
        euler_formula = Text("e^ix = cos(x) + i sin(x)", color=WHITE)
        self.place_in_area(euler_formula, "A2", "A6", scale_factor=0.7)
        
        # Coordinate Plane - B2-E6 area (middle right)
        plane = NumberPlane(
            x_range=[-1.5, 1.5, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=3,
            y_length=3,
            axis_config={"stroke_opacity": 0.5},
            background_line_style={"stroke_opacity": 0.2}
        )
        unit_circle = Circle(radius=plane.get_x_unit_size(), color=WHITE, stroke_opacity=0.6)
        
        # Grouping to use grid positioning correctly
        viz_group = VGroup(plane, unit_circle)
        self.place_in_area(viz_group, "B2", "E6", scale_factor=0.8)
        
        self.play(Write(euler_formula), run_time=1)
        self.play(Create(plane), Create(unit_circle), run_time=1)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Highlight line 2 in green
        self.play(self.lecture[1].animate.set_color("#00FF00"), run_time=0.5)
        
        x_tracker = ValueTracker(0)
        
        # Vector at angle x in green (#00FF00)
        vector = Line(
            start=plane.coords_to_point(0,0),
            end=plane.coords_to_point(1,0),
            color="#00FF00",
            stroke_width=6
        )
        
        # Label for angle x
        x_label = Text("x", color="#00FF00", font_size=20)
        
        # Efficient updaters
        def update_vector(v):
            v.put_start_and_end_on(
                plane.coords_to_point(0,0),
                plane.coords_to_point(np.cos(x_tracker.get_value()), np.sin(x_tracker.get_value()))
            )
        
        def update_x_label(l):
            l.move_to(plane.coords_to_point(
                1.2 * np.cos(x_tracker.get_value()), 
                1.2 * np.sin(x_tracker.get_value())
            ))
            
        vector.add_updater(update_vector)
        x_label.add_updater(update_x_label)
        
        self.add(vector, x_label)
        self.play(x_tracker.animate.set_value(PI/3), run_time=1.2)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Highlight line 3 in a light blue shade
        self.play(self.lecture[2].animate.set_color("#87CEEB"), run_time=0.5)
        
        # Projections: Cosine (red) and Sine (blue)
        cos_line = Line(color="#FF0000", stroke_width=8)
        cos_line.add_updater(lambda cl: cl.put_start_and_end_on(
            plane.coords_to_point(0,0),
            plane.coords_to_point(np.cos(x_tracker.get_value()), 0)
        ))
        
        sin_line = Line(color="#0000FF", stroke_width=8)
        sin_line.add_updater(lambda sl: sl.put_start_and_end_on(
            plane.coords_to_point(np.cos(x_tracker.get_value()), 0),
            plane.coords_to_point(np.cos(x_tracker.get_value()), np.sin(x_tracker.get_value()))
        ))
        
        self.play(Create(cos_line), Create(sin_line), run_time=1)
        
        # Move to PI to show transition to Euler's Identity
        self.play(x_tracker.animate.set_value(PI), run_time=1.8)
        
        # Euler's Identity result at bottom - area F2-F6
        # Fixed: Switched from MathTex to Text and used Unicode for Pi symbol
        euler_identity = Text("e^iπ = -1", color=GOLD)
        self.place_in_area(euler_identity, "F2", "F6", scale_factor=0.7)
        
        self.play(Write(euler_identity), run_time=1)
        self.play(Indicate(euler_identity), run_time=1)
        self.wait(1.5)

if __name__ == "__main__":
    pass
