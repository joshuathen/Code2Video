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
        # Mandatory call to setup_layout with section title and lecture lines
        self.setup_layout(
            "Visualizing the Slopes on a Curve", 
            [
                "Look at the circle at point (3, 4).", 
                "Our formula gives a slope of negative three-fourths.", 
                "Notice how the tangent line matches this visual tilt."
            ]
        )

        # === Preparation of Mobjects ===
        # Use Axes to establish a coordinate system within the grid area (A1 to F6)
        # The right-side area is approx 5x5 units in scene space.
        axes = Axes(
            x_range=[-6, 6, 2],
            y_range=[-6, 6, 2],
            x_length=4.5,
            y_length=4.5,
            axis_config={"color": WHITE, "include_tip": True}
        )
        
        # Circle x^2 + y^2 = 25 -> radius 5 on axes.
        # We calculate scene radius relative to the axes scale.
        unit_size = axes.coords_to_point(1, 0)[0] - axes.coords_to_point(0, 0)[0]
        circle = Circle(radius=5 * unit_size, color="#FFFF00") 
        
        # Point (3, 4) in red
        dot_pos = axes.coords_to_point(3, 4)
        dot = Dot(point=dot_pos, color="#FF0000")
        dot_label = Text("(3, 4)", font_size=20, color="#FF0000")
        
        # Tangent line: dy/dx = -x/y = -3/4 at (3, 4)
        # Equation: y - 4 = -0.75(x - 3)  => y = -0.75x + 6.25
        # Define line using two points on the axes
        p1 = axes.coords_to_point(-1, 7)
        p2 = axes.coords_to_point(7, 1)
        tangent_line = Line(p1, p2, color="#00FF00")
        
        # Slope label text
        slope_label = Text("slope = -3/4", font_size=22, color="#FFFFFF")
        
        # Visual Anchor System: Positioning
        # Group the mathematical objects for relative positioning
        math_group = VGroup(axes, circle, dot, tangent_line)
        # Place the entire graph in the right-side grid area
        self.place_in_area(math_group, 'A1', 'F6')
        
        # Position individual labels using specific grid points
        # Point (3, 4) is roughly in the B5/A5 area of the scene
        self.place_at_grid(dot_label, 'A5', scale_factor=0.8)
        # Slope label near the tangent line in the lower right area
        self.place_at_grid(slope_label, 'C6', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # Highlight first line in yellow to match circle
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.play(Create(axes), Create(circle), run_time=1.5)
        self.play(FadeIn(dot), Write(dot_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line is already white, keeping focus color consistent with slope_label
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(WHITE)
        )
        self.play(Write(slope_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third line in green to match tangent line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FF00")
        )
        self.play(Create(tangent_line))
        self.wait(3)
