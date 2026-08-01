from manim import *
import numpy as np

# Use the provided TeachingScene base class without modification.
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

class Section6Scene(TeachingScene):
    def construct(self):
        # Metadata retrieved from shared state
        title_text = "Summary: Calculus in the Real World"
        lecture_lines = [
            "Calculus models everything from rocket launches to viruses.",
            "It transforms frozen snapshots into a moving reality.",
            "Master change and you master the language of nature."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Hexadecimal colors (L008)
        COLOR_ROCKET = "#FFFFFF"
        COLOR_TRAJECTORY = "#00FFFF"
        COLOR_VELOCITY = "#FF8000"
        COLOR_ORBIT = "#FFFF00"
        COLOR_CALCULUS = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Show a white #FFFFFF rocket launching and following a cyan #00FFFF parabolic trajectory.
        self.play(self.lecture[0].animate.set_color(COLOR_ROCKET))
        
        # Simple Rocket mobject: Body and Nose
        rocket_body = Rectangle(width=0.2, height=0.4, color=COLOR_ROCKET, fill_opacity=1)
        rocket_nose = Triangle(color=COLOR_ROCKET, fill_opacity=1).scale(0.12).next_to(rocket_body, UP, buff=0)
        rocket = VGroup(rocket_body, rocket_nose)
        
        # Issue 34: Changed starting position to F5 to avoid lecture notes (L002)
        self.place_at_grid(rocket, "F5", scale_factor=0.6)
        
        # Target orbit configuration
        # Centering at B3 ensures that when the rocket enters at B4 (3 o'clock), 
        # the path is diagonal and avoids the lecture text on the left.
        orbit_center_pos = self.grid["B3"]
        orbit_radius = 1.0
        orbit_circle = Circle(radius=orbit_radius, color=COLOR_ORBIT).move_to(orbit_center_pos)
        orbit_entry = orbit_circle.point_from_proportion(0) # This point is B4
        
        start_pt = self.grid["F5"]
        # Approximate a parabola with an arc for smooth motion
        trajectory = ArcBetweenPoints(start_pt, orbit_entry, angle=-TAU/8).set_color(COLOR_TRAJECTORY)
        
        # Orient rocket for launch direction
        rocket.rotate(-45 * DEGREES) 
        
        self.play(Create(trajectory), run_time=1.5)
        # Rocket follows the arc path
        self.play(MoveAlongPath(rocket, trajectory), run_time=2.5, rate_func=smooth)
        self.wait(2.0) 

        # === Animation for Lecture Line 2 ===
        # Flash the word 'Velocity' in orange #FF8000 at multiple points along the trajectory.
        self.play(self.lecture[1].animate.set_color(COLOR_VELOCITY))
        
        v_labels = VGroup(
            Text("Velocity", font_size=20, color=COLOR_VELOCITY),
            Text("Velocity", font_size=20, color=COLOR_VELOCITY),
            Text("Velocity", font_size=20, color=COLOR_VELOCITY)
        )
        
        # Position labels following Proximity Rule (L002)
        proportions = [0.2, 0.5, 0.8]
        for i, prop in enumerate(proportions):
            pt = trajectory.point_from_proportion(prop)
            # Offset to avoid overlapping the trajectory line
            v_labels[i].move_to(pt + RIGHT * 0.8 + UP * 0.2)
            self.play(FadeIn(v_labels[i]))
            self.play(Indicate(v_labels[i], color=COLOR_VELOCITY)) # L004: Visual highlighting
        
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # The rocket reaches a circular yellow #FFFF00 orbit while the word 'Calculus' appears in glowing white #FFFFFF.
        self.play(self.lecture[2].animate.set_color(COLOR_ORBIT))
        
        # Clear previous markers
        self.play(FadeOut(trajectory), FadeOut(v_labels))
        
        # Move rocket to the orbital entry point
        self.play(
            rocket.animate.move_to(orbit_entry).rotate(45 * DEGREES), # Re-orient to "up"
            Create(orbit_circle),
            run_time=2
        )
        
        # Centerpiece text
        calculus_label = Text("Calculus", font_size=36, color=COLOR_CALCULUS)
        # Issue 35: Positioned at B5 and scaled to 0.7 to avoid overlap with orbit center B3/path.
        self.place_at_grid(calculus_label, "B5", scale_factor=0.7)
        
        # Low-complexity updater for orbital motion (L027)
        angle_tracker = ValueTracker(0)
        rocket.add_updater(lambda m: m.move_to(orbit_circle.point_from_proportion(angle_tracker.get_value() % 1)))
        
        self.play(FadeIn(calculus_label), Indicate(calculus_label, color=COLOR_CALCULUS))
        self.play(angle_tracker.animate.set_value(1.5), run_time=4, rate_func=linear)
        
        # Clean up and finalize
        rocket.clear_updaters()
        self.wait(2.0)
