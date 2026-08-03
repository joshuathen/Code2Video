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
        self.setup_layout("Unfolding the Circle", [
            "Increasing mass ratio makes each step much smaller.",
            "More collisions are needed to traverse the circle.",
            "The total count relates to the circle's circumference.",
            "Pi emerges from the ratio of distance to diameter.",
            "The number of steps reveals Pi's hidden digits."
        ])

        # Colors
        YELLOW_COLOR = "#FFFF00"
        GREEN_COLOR = "#00FF00"
        WHITE_COLOR = "#FFFFFF"
        MAGENTA_COLOR = "#FF00FF"

        # === Animation for Lecture Line 1 ===
        # Highlight a small yellow arc segment (#FFFF00) representing the change from one collision.
        self.lecture[0].set_color(YELLOW_COLOR)
        
        base_circle = Arc(radius=1.5, start_angle=0, angle=TAU, color=GREEN_COLOR)
        # Position the circle first so we can place elements relative to it
        self.place_in_area(base_circle, "B2", "D5")
        
        # Creating small_arc relative to base_circle
        small_arc = Arc(radius=1.5, start_angle=PI/4, angle=0.4, color=YELLOW_COLOR, stroke_width=8)
        small_arc.move_to(base_circle.get_center() + np.array([1.5*np.cos(PI/4+0.2), 1.5*np.sin(PI/4+0.2), 0]))

        self.play(Create(base_circle))
        self.play(Create(small_arc))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show the arc segment shrinking as the mass ratio M/m is increased.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW_COLOR)

        smaller_arc = Arc(radius=1.5, start_angle=PI/4, angle=0.1, color=YELLOW_COLOR, stroke_width=8)
        smaller_arc.move_to(base_circle.get_center() + np.array([1.5*np.cos(PI/4+0.05), 1.5*np.sin(PI/4+0.05), 0]))

        # Create more arcs to show increased collisions
        more_arcs = VGroup(*[
            Arc(radius=1.5, start_angle=PI/4 + i*0.15, angle=0.1, color=YELLOW_COLOR, stroke_width=4)
            for i in range(1, 10)
        ])
        for arc in more_arcs:
            # Shift each arc to the circle's position
            arc.move_to(base_circle.get_center() + (arc.get_center() - ORIGIN))

        self.play(Transform(small_arc, smaller_arc))
        self.play(Create(more_arcs))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate the entire green circle (#00FF00) unrolling into a straight white line (#FFFFFF).
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN_COLOR)

        self.play(FadeOut(small_arc), FadeOut(more_arcs))

        # ISSUE 34: Move unrolled_line up to E1-E6 for breathing room
        unrolled_line = Line(start=LEFT*2.5, end=RIGHT*2.5, color=WHITE_COLOR)
        self.place_in_area(unrolled_line, "E1", "E6")

        # Unrolling effect: circle moves and "straightens"
        self.play(
            base_circle.animate.rotate(-TAU).move_to(unrolled_line.get_center()),
            ReplacementTransform(base_circle, unrolled_line),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Fill the straight line with these small arc segments to count how many fit.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(WHITE_COLOR)

        num_segments = 31 # Suggesting pi
        line_segments = VGroup(*[
            Line(
                start=unrolled_line.get_start() + RIGHT*(i/num_segments)*5,
                end=unrolled_line.get_start() + RIGHT*((i+0.8)/num_segments)*5,
                color=YELLOW_COLOR,
                stroke_width=4
            )
            for i in range(num_segments)
        ])

        self.play(Create(line_segments, lag_ratio=0.05))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The total count of segments morphs into the decimal digits of Pi (#FF00FF)
        # while incorporating the [Asset: ...] icon.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(MAGENTA_COLOR)

        # ISSUE 35 & 36: pi_text at D3, scale_factor=1.2
        pi_text = Text("3.14159...", font_size=48, color=MAGENTA_COLOR)
        self.place_at_grid(pi_text, "D3", scale_factor=1.2)

        # ISSUE 21: Asset integration
        # Load the icon from the specified path
        based_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/based.svg")
        # Place it near pi_text, at D5
        self.place_at_grid(based_icon, "D5", scale_factor=0.8)

        self.play(
            ReplacementTransform(line_segments, pi_text),
            FadeIn(based_icon),
            unrolled_line.animate.set_stroke(opacity=0.3)
        )
        self.wait(2)
