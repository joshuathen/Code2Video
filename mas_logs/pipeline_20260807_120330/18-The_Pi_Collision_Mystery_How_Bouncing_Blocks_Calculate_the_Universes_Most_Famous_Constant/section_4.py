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
        title = "The Geometry of a Collision"
        lecture_lines = [
            "Each collision is a jump on the circle's edge.",
            "Wall bounces and block hits follow strict geometric paths.",
            "This creates a series of equal-sized arcs.",
            "Imagine the system as a ball bouncing in a wedge.",
            "The wedge angle shrinks as the mass ratio grows."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        COLOR_CIRCLE = "#00FFFF"
        COLOR_REFLECTION = "#ADD8E6"
        COLOR_ARC = "#FFA500"
        COLOR_WEDGE = "#FFFFFF"
        COLOR_POINT = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # "Each collision is a jump on the circle's edge."
        self.play(self.lecture[0].animate.set_color(COLOR_CIRCLE))
        
        # Create the cyan circle
        # B021: Leave Column 1 mostly for text, so visuals start from Column 2/3
        circle = Circle(radius=1.5, color=COLOR_CIRCLE)
        # Issue 31: Fix circle area to B2-E4 (leaving room for labels on right)
        self.place_in_area(circle, "B2", "E4")
        center = circle.get_center()
        
        # Dot representing the state (v1, v2)
        state_dot = Dot(circle.point_at_angle(PI/4), color=COLOR_POINT)
        
        self.play(Create(circle))
        self.play(FadeIn(state_dot))
        
        # Simulate a jump
        target_point = circle.point_at_angle(3*PI/4)
        self.play(state_dot.animate.move_to(target_point), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Wall bounces and block hits follow strict geometric paths."
        self.play(self.lecture[1].animate.set_color(COLOR_REFLECTION))
        
        # Reflection line (e.g., vertical for wall hits in this space)
        reflection_line = DashedLine(
            start=center + UP * 1.8, 
            end=center + DOWN * 1.8, 
            color=COLOR_REFLECTION
        )
        
        # Reflection logic visualization
        self.play(Create(reflection_line))
        
        # Mirror jump
        start_pt = circle.point_at_angle(PI/6)
        end_pt = circle.point_at_angle(5*PI/6)
        state_dot.move_to(start_pt)
        self.play(state_dot.animate.move_to(end_pt), run_time=1, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "This creates a series of equal-sized arcs."
        self.play(self.lecture[2].animate.set_color(COLOR_ARC))
        
        # Fixed orange arc
        arc_angle = PI/6
        highlight_arc = Arc(
            radius=1.5, 
            start_angle=PI/4, 
            angle=arc_angle, 
            color=COLOR_ARC, 
            stroke_width=6
        )
        highlight_arc.move_to(center)
        
        arc_label = Text("Fixed Arc", font_size=18, color=COLOR_ARC)
        # Issue 32: Place label at C5 for better proximity to the visual
        self.place_at_grid(arc_label, "C5", scale_factor=0.8)
        
        self.play(Create(highlight_arc), FadeIn(arc_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Imagine the system as a ball bouncing in a wedge."
        self.play(self.lecture[3].animate.set_color(COLOR_WEDGE))
        
        # Wedge lines
        wedge_angle = PI/3
        line1 = Line(center, center + 2.5 * RIGHT, color=COLOR_WEDGE)
        line2 = Line(center, center + 2.5 * rotate_vector(RIGHT, wedge_angle), color=COLOR_WEDGE)
        wedge = VGroup(line1, line2)
        
        # Hide previous elements to clarify the wedge metaphor
        self.play(
            circle.animate.set_stroke(opacity=0.3),
            reflection_line.animate.set_stroke(opacity=0.3),
            highlight_arc.animate.set_stroke(opacity=0.3),
            FadeOut(state_dot),
            FadeOut(arc_label)
        )
        
        self.play(Create(wedge))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "The wedge angle shrinks as the mass ratio grows."
        self.play(self.lecture[4].animate.set_color(COLOR_WEDGE))
        
        # Animate wedge shrinking
        new_angle = PI/12
        new_line2 = Line(center, center + 2.5 * rotate_vector(RIGHT, new_angle), color=COLOR_WEDGE)
        
        # Label for wedge angle
        angle_label = MathTex(r"\theta \propto \sqrt{m/M}", font_size=20, color=COLOR_WEDGE)
        # Issue 33: Place label at D5 to avoid extreme right edge
        self.place_at_grid(angle_label, "D5", scale_factor=0.7)
        
        self.play(
            Transform(line2, new_line2),
            FadeIn(angle_label)
        )
        
        self.wait(2)
