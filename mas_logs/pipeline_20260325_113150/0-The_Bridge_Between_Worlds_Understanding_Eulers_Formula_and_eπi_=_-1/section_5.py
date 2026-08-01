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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup the layout with title and lecture lines
        self.setup_layout(
            "The Journey to \u03c0: Why e^(\u03c0i) = -1", 
            [
                "Let's set theta to pi, exactly half a rotation.", 
                "We travel halfway around the edge of the circle.", 
                "We land perfectly on the real value negative one."
            ]
        )

        # Calculate central origin for the animation area (B2 to E5)
        # B2: i=1, j=1 -> x=1.5, y=1.2
        # E5: i=4, j=4 -> x=4.5, y=-1.8
        tl_pos = self.grid["B2"]
        br_pos = self.grid["E5"]
        origin = (tl_pos + br_pos) / 2

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        # Create Axes and Circle
        axes = Axes(
            x_range=[-1.2, 1.2], 
            y_range=[-1.2, 1.2], 
            x_length=3.6, 
            y_length=3.6, 
            axis_config={"include_tip": True, "color": WHITE}
        )
        circle = Circle(radius=1.5, color=WHITE)
        
        # Place the coordinate plane components
        plane_group = VGroup(axes, circle)
        self.place_in_area(plane_group, "B2", "E5")
        
        # Re-derive origin after placement to ensure absolute accuracy
        origin = axes.get_origin()
        
        # Vector at theta=0 (pointing to '1')
        vector = Arrow(origin, origin + RIGHT * 1.5, color="#FFFF00", buff=0)
        
        # Label '1' at grid D5 (Near the right edge of the circle)
        label_1 = Text("1", font_size=24, color=WHITE)
        self.place_at_grid(label_1, "D5")
        
        # Initial partial equation (Using MarkupText to avoid LaTeX dependency)
        eq = MarkupText("e<sup>iπ</sup>", font_size=36, color=WHITE)
        self.place_in_area(eq, "A3", "A5")
        
        self.play(Create(axes), Create(circle))
        self.play(GrowArrow(vector), Write(label_1), FadeIn(eq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        # Path trace arc from 0 to pi
        arc = Arc(radius=1.5, start_angle=0, angle=PI, arc_center=origin, color="#00FFFF")
        
        # Animate the rotation and the arc trace
        self.play(
            Rotate(vector, angle=PI, about_point=origin),
            Create(arc),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        
        # Label '-1' at grid D2 (Near the left edge of the circle)
        label_minus_1 = Text("-1", font_size=28, color="#FF0000")
        self.place_at_grid(label_minus_1, "D2")
        
        # Final completed equation
        eq_full = MarkupText("e<sup>iπ</sup> = -1", font_size=36, color=WHITE)
        self.place_in_area(eq_full, "A3", "A5")
        
        # Point of impact on the Real axis
        dot = Dot(point=origin + LEFT * 1.5, color="#FF0000")
        
        self.play(
            Write(label_minus_1),
            FadeIn(dot),
            Flash(dot, color="#FF0000", line_length=0.3),
            Transform(eq, eq_full)
        )
        self.wait(2)
