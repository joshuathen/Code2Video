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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup the layout with the section title and lecture lines from storyboard
        title = "Geometric Intuition: The 'Infinite Zoom'"
        lines = [
            "Zooming in on a curve reveals a surprising secret.",
            "At extreme magnification, every smooth curve looks straight.",
            "This straight line is the tangent at that point.",
            "The derivative is the slope of this tangent line.",
            "It captures the exact rate of change right now."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Draw a red parabola (#FF0000) and mark a single point with a white dot (#FFFFFF).
        self.play(self.lecture[0].animate.set_color(RED))
        
        # Define the parabola y = 0.5 * x^2
        parabola = FunctionGraph(lambda x: 0.5 * x**2, x_range=[-2.5, 2.5]).set_color("#FF0000")
        
        # Point on the curve at x=1.0, y=0.5
        dot = Dot(color="#FFFFFF")
        # Fix for Issue 30: Move dot to C4 to avoid obstructing lecture lines
        self.place_at_grid(dot, 'C4')
        
        # Shift parabola so that its point (1.0, 0.5) aligns with the dot's grid position
        # FunctionGraph relative coordinate for the point
        point_on_curve_rel = np.array([1.0, 0.5, 0])
        parabola.shift(dot.get_center() - point_on_curve_rel)
        
        self.play(Create(parabola), FadeIn(dot))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Create a magnifying glass circle around the white dot.
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        mag_glass = Circle(radius=0.7, color=WHITE, stroke_width=4)
        mag_glass.move_to(dot.get_center())
        
        self.play(Create(mag_glass))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Zoom into the area inside the circle. The curve should gradually look straight.
        # Overlay a green tangent line (#00FF00).
        self.play(self.lecture[2].animate.set_color(GREEN))
        
        zoom_group = VGroup(parabola, dot)
        
        # Tangent line at x=1 for y=0.5x^2 has slope f'(1)=1.0
        tangent_line = Line(start=LEFT*2.5, end=RIGHT*2.5, color="#00FF00")
        tangent_line.rotate(np.arctan(1.0)) 
        tangent_line.move_to(dot.get_center())
        
        # Scale the parabola and dot around the point to simulate zoom
        self.play(
            zoom_group.animate.scale(8, about_point=dot.get_center()),
            parabola.animate.set_stroke(width=2),
            mag_glass.animate.scale(1.2).set_color(GRAY), # Fade/expand the glass slightly
            run_time=3
        )
        
        self.play(Create(tangent_line))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Display 'Instantaneous Slope' in white (#FFFFFF)
        self.play(self.lecture[3].animate.set_color(WHITE))
        
        slope_label = Text("Instantaneous Slope", font_size=20, color="#FFFFFF")
        # Fix for Issue 31: Use place_in_area to avoid overlapping tangent line
        self.place_in_area(slope_label, 'E4', 'F6', scale_factor=0.8)
        
        self.play(Write(slope_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Focus on rate of change
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        # Highlight the point of tangency
        self.play(
            Indicate(dot, color=WHITE, scale_factor=1.5),
            Flash(dot, color=WHITE, line_length=0.3)
        )
        self.wait(2)
