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
        # Section Title and Lecture Lines
        title = "The Big Picture: Static vs. Dynamic"
        lines = [
            "Algebra handles constant speeds and steady rates perfectly.",
            "But the real world changes at every single moment.",
            "Calculus is the mathematics of change and accumulation."
        ]
        self.setup_layout(title, lines)
        
        # Dim all lecture lines initially for highlighting effect
        for line in self.lecture:
            line.set_color("#777777")

        # === Animation for Lecture Line 1 ===
        # Highlight first line (White for Algebra/Static)
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # Create a white straight line ramp
        ramp = Line(start=LEFT*1.5, end=RIGHT*1.5, color="#FFFFFF")
        # Place ramp in the designated area (Issue 22: B3 to C6)
        self.place_in_area(ramp, "B3", "C6")
        
        # 'Static' label positioned near the ramp (Issue 21: A4)
        static_label = Text("Static", font_size=24, color="#FFFFFF")
        self.place_at_grid(static_label, "A4", scale_factor=0.8)
        
        self.play(Create(ramp), Write(static_label))
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # Dim first line, highlight second line (Green for Dynamic)
        self.play(
            self.lecture[0].animate.set_color("#777777"),
            self.lecture[1].animate.set_color("#00FF00")
        )
        
        # Morph the straight line into a green curve
        # We'll create a green line in the bottom area and morph it to a curve
        # Issue 23: E3 to F6
        curve_base = Line(start=LEFT*1.5, end=RIGHT*1.5, color="#00FF00")
        self.place_in_area(curve_base, "E3", "F6")
        
        # Define the target curve path (using a smooth VMobject)
        curve_path = VMobject(color="#00FF00")
        # Creating a parabolic shape
        points = [LEFT*1.5 + DOWN*0.2, ORIGIN + UP*0.5, RIGHT*1.5 + DOWN*0.2]
        curve_path.set_points_as_corners(points)
        curve_path.make_smooth()
        self.place_in_area(curve_path, "E3", "F6")
        
        # Show the transition from static (straight line) to dynamic (curve)
        self.play(Create(curve_base))
        self.play(ReplacementTransform(curve_base, curve_path), run_time=2)
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # Dim second line, highlight third line (Yellow for Calculus)
        self.play(
            self.lecture[1].animate.set_color("#777777"),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        
        # Create moving objects: Turtle (constant) and Puppy (variable)
        # Issue 19: Use Asset for turtle
        turtle_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/turtle.svg"
        turtle = SVGMobject(turtle_asset_path).set_color("#FFFFFF").scale(0.3)
        puppy = Dot(color="#00FF00")
        
        # Scaled labels for the dots (L002)
        turtle_label = Text("Turtle", font_size=20, color="#FFFFFF").scale(0.8)
        puppy_label = Text("Puppy", font_size=20, color="#00FF00").scale(0.8)
        
        # Initial positions at the start of their respective paths
        turtle.move_to(ramp.point_from_proportion(0))
        puppy.move_to(curve_path.point_from_proportion(0))
        
        # Updaters for labels to follow the mobjects (Proximity Rule L002)
        turtle_label.add_updater(lambda m: m.next_to(turtle, UP, buff=0.1))
        puppy_label.add_updater(lambda m: m.next_to(puppy, UP, buff=0.1))
        
        self.add(turtle, puppy, turtle_label, puppy_label)
        
        # Animation tracker for the "race"
        tracker = ValueTracker(0)
        
        # Turtle moves at a constant speed (proportional to tracker)
        turtle.add_updater(lambda m: m.move_to(ramp.point_from_proportion(tracker.get_value())))
        
        # Puppy moves at a variable speed (accelerating)
        # Using tracker^1.5 for non-linear movement
        puppy.add_updater(lambda m: m.move_to(curve_path.point_from_proportion(tracker.get_value()**1.5)))
        
        # Execute the race (4 seconds for clarity)
        # Using rate_functions.linear as per L024
        self.play(tracker.animate.set_value(1), run_time=4, rate_func=rate_functions.linear)
        self.wait(2.0)
