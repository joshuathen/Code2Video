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
        # Data from storyboard
        title_text = "The Solution: The Cycloid"
        lecture_lines = [
            "The fastest path is an inverted cycloid curve.",
            "A cycloid is traced by a point on a rolling wheel.",
            "Its parametric equations balance path length and acceleration.",
            "Gravity accelerates the object most efficiently along this arc.",
            "This curve is the unique solution to the Brachistochrone problem."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        
        # Define the inverted cycloid path
        radius = 0.5
        # Start and end points for visual consistency
        p_start = self.grid["B2"]
        p_end = self.grid["F6"]
        
        # Parametric cycloid: x = r(t - sin t), y = -r(1 - cos t)
        # We'll adjust it to fit the right-side workspace
        cycloid_func = ParametricFunction(
            lambda t: np.array([
                radius * (t - np.sin(t)),
                -radius * (1 - np.cos(t)),
                0
            ]),
            t_range=[0, TAU],
            color="#00FFFF"
        )
        self.place_in_area(cycloid_func, "B2", "F6", scale_factor=1.2) # Issue 34/35 Fix
        
        self.play(Create(cycloid_func), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF0000") # Red color for dot
        
        # Asset: Wheel
        wheel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wheel.svg")
        wheel.set_color(WHITE)
        wheel_radius = 0.4
        wheel.height = wheel_radius * 2
        
        # Setup rolling animation
        # We need a flat line to roll along
        ceiling = Line(self.grid["A1"], self.grid["A6"], color=WHITE)
        wheel_start_center = np.array([self.grid["A1"][0], self.grid["A1"][1] - wheel_radius, 0])
        wheel.move_to(wheel_start_center)
        
        dot = Dot(radius=0.08, color="#FF0000")
        # Dot starts at contact point with ceiling (top of wheel)
        dot.move_to(wheel.get_top())
        
        theta_tracker = ValueTracker(0)
        
        def update_wheel(mob):
            theta = theta_tracker.get_value()
            mob.move_to(wheel_start_center + np.array([wheel_radius * theta, 0, 0]))
            mob.set_rotation(-theta)
            
        def update_dot(mob):
            theta = theta_tracker.get_value()
            center = wheel_start_center + np.array([wheel_radius * theta, 0, 0])
            # Rotation of dot around center
            # At theta=0, relative pos is (0, r, 0)
            rel_pos = np.array([
                wheel_radius * np.sin(theta),
                wheel_radius * np.cos(theta),
                0
            ])
            mob.move_to(center + rel_pos)

        wheel.add_updater(update_wheel)
        dot.add_updater(update_dot)
        
        # Tracing the path
        trace = TracedPath(dot.get_center, stroke_color="#00FFFF", stroke_width=4)
        
        self.play(Create(ceiling), FadeIn(wheel), FadeIn(dot))
        self.add(trace)
        
        # Roll for one full rotation
        self.play(theta_tracker.animate.set_value(TAU), run_time=4, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(WHITE) # Text asks for white next to wheel
        
        # Show parametric equations in white
        eq1 = MathTex("x = r(\\theta - \\sin \\theta)", color=WHITE, font_size=24)
        eq2 = MathTex("y = r(1 - \\cos \\theta)", color=WHITE, font_size=24)
        eq_group = VGroup(eq1, eq2).arrange(DOWN, aligned_edge=LEFT)
        
        # Place next to wheel (Issue 26)
        self.place_at_grid(eq_group, "C5", scale_factor=1.0)
        
        self.play(Write(eq_group))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FFFF00") # Yellow pulse
        
        # Clear wheel and ceiling
        wheel.remove_updater(update_wheel)
        dot.remove_updater(update_dot)
        self.play(FadeOut(wheel), FadeOut(dot), FadeOut(ceiling), FadeOut(trace), FadeOut(eq_group))
        
        # Steepest segment highlight (beginning of the cycloid)
        # Create a partial cycloid for the pulse
        steep_part = ParametricFunction(
            lambda t: np.array([
                radius * (t - np.sin(t)),
                -radius * (1 - np.cos(t)),
                0
            ]),
            t_range=[0, PI/2],
            color="#FFFF00"
        )
        self.place_in_area(steep_part, "B2", "F6", scale_factor=1.2)
        
        self.play(Indicate(steep_part, color="#FFFF00", scale_factor=1.1))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFD700") # Gold label
        
        label = Text("Cycloid", color="#FFD700", font_size=32)
        label.next_to(cycloid_func, DOWN, buff=0.2)
        
        self.play(FadeIn(label, shift=UP))
        self.wait(3)

# Issue Updates
# update_issue(26, under_review=True, resolution_note="Integrated SVG wheel asset and displayed parametric equations in white.")
# update_issue(34, under_review=True, resolution_note="Adjusted slide_version scaling to 1.2 and moved to B2-F6 to avoid overlap.")
# update_issue(35, under_review=True, resolution_note="Improved vertical placement by shifting to B2-F6 area.")
