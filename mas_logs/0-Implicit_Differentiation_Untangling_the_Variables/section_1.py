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
        # Setup layout with title and lecture lines
        self.setup_layout("The Wall: Explicit vs. Implicit", [
            "Meet explicit functions, where y is clearly isolated.",
            "But what if y is trapped inside the equation?",
            "We call these \"implicit\" equations, like a locked box.",
            "Consider this circular force field: x² plus y² equals 25.",
            "Finding the slope here requires a new mathematical tool."
        ])

        # === Animation for Lecture Line 1 ===
        # "Meet explicit functions, where y is clearly isolated."
        self.lecture[0].set_color(WHITE)
        explicit_eq = Text("y = x²", color=WHITE)
        self.place_at_grid(explicit_eq, "A3", scale_factor=0.8)
        self.play(Write(explicit_eq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "But what if y is trapped inside the equation?"
        self.lecture[1].set_color(WHITE)
        # Display the implicit equation below the explicit one
        # Using a VGroup of Text to isolate 'y' for the locked box later
        implicit_eq = VGroup(
            Text("x² + ", color="#00FFFF"),
            Text("y", color="#00FFFF"),
            Text("² = 25", color="#00FFFF")
        ).arrange(RIGHT, buff=0.1)
        self.place_at_grid(implicit_eq, "B3", scale_factor=0.8)
        self.play(FadeIn(implicit_eq))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "We call these \"implicit\" equations, like a locked box."
        self.lecture[2].set_color(RED)
        # Animate a red locked box appearing around the 'y' variable
        locked_box = SurroundingRectangle(implicit_eq[1], color=RED, buff=0.1)
        self.play(Create(locked_box))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Consider this circular force field: x² plus y² equals 25."
        self.lecture[3].set_color("#00FFFF")
        
        # Circle radius to fit in the grid area C2-F5
        circle = Circle(radius=1.5, color="#00FFFF")
        self.place_in_area(circle, "C2", "F5")
        
        # Moving dot representing a drone on the circle
        theta_tracker = ValueTracker(0)
        drone = Dot(color="#00FFFF")
        drone.add_updater(lambda d: d.move_to(circle.point_at_angle(theta_tracker.get_value())))
        
        self.play(Create(circle))
        self.add(drone)
        self.play(theta_tracker.animate.set_value(PI/4), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Finding the slope here requires a new mathematical tool."
        self.lecture[4].set_color(YELLOW)
        
        # Yellow tangent line showing the slope
        def get_tangent_line():
            angle = theta_tracker.get_value()
            pos = circle.point_at_angle(angle)
            # Tangent direction vector for a circle: (-sin(theta), cos(theta))
            tangent_direction = np.array([-np.sin(angle), np.cos(angle), 0])
            return Line(
                pos - tangent_direction * 1.0, 
                pos + tangent_direction * 1.0, 
                color=YELLOW,
                stroke_width=4
            )

        tangent = always_redraw(get_tangent_line)
        self.play(Create(tangent))
        # Animate the drone and tangent moving along the circular path
        self.play(theta_tracker.animate.increment_value(PI), run_time=3, rate_func=linear)
        self.wait(2)
