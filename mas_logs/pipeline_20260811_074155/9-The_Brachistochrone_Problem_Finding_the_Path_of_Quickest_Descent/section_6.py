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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Tautochrone Property (Bonus)", [
            "The cycloid also possesses the amazing tautochrone property.",
            "Objects released from any height reach bottom simultaneously.",
            "Descent time is independent of the starting position."
        ])
        
        # Parameters for the cycloid
        r = 0.5
        def cycloid_func(theta):
            # Inverted cycloid centered such that bottom is (0,0)
            return np.array([
                r * (theta + np.sin(theta)),
                r * (1 - np.cos(theta)),
                0
            ])

        # Create the cycloid path
        # The parametric curve is generated from -PI to PI
        cycloid_path = ParametricFunction(cycloid_func, t_range=[-PI, PI], color=GRAY)
        # Resolved Issue 33: Reduced scale factor from 1.5 to 0.85 to avoid occlusion
        self.place_in_area(cycloid_path, 'A1', 'F6', scale_factor=0.85)
        
        # Positioning logic for points on the path
        # In local coordinates, the curve's bounding box center is at (0, r)
        # because x goes from -r*PI to r*PI and y from 0 to 2r.
        def get_cycloid_pos(theta):
            # Local coordinate of the point on the base curve
            local_p = cycloid_func(theta)
            # Center of the base curve bounding box
            local_center = np.array([0, r, 0])
            # Vector from center to point, scaled by the same factor as the path (0.85)
            scaled_vec = (local_p - local_center) * 0.85
            # Add to the world position of the mobject's center
            return cycloid_path.get_center() + scaled_vec

        # Load marble asset - Resolved Issue 23
        marble_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/mar.svg"
        # Pre-scaling the asset
        marble_master = SVGMobject(marble_path).scale(0.15)

        # === Animation for Lecture Line 1 ===
        # "The cycloid also possesses the amazing tautochrone property."
        self.lecture[0].set_color(YELLOW)
        self.play(Create(cycloid_path))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Objects released from any height reach bottom simultaneously."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Initial positions for marbles (starting at different heights on the left side)
        theta_starts = [-0.8 * PI, -0.5 * PI, -0.2 * PI]
        colors = ["#FF0000", "#00FF00", "#0000FF"]
        marbles = VGroup()
        for theta, color in zip(theta_starts, colors):
            # Using the asset for marbles as per Issue 23
            marble = marble_master.copy()
            marble.set_color(color)
            marble.move_to(get_cycloid_pos(theta))
            marbles.add(marble)
        
        self.play(FadeIn(marbles))
        self.wait(1)

        # Time tracker for the descent
        # Based on the tautochrone property: arc length s(t) = s0 * cos(omega * t)
        # For the cycloid, s is proportional to sin(theta/2).
        T_descent = 2.0
        # Descent to bottom occurs at cos(omega*t) = 0, so omega*T_descent = PI/2
        omega = PI / (2 * T_descent)
        
        time_tracker = ValueTracker(0)
        
        # Add updaters for marbles
        def get_updater(t0):
            # Uses s(t) = s0 * cos(omega * t) relationship
            return lambda m: m.move_to(
                get_cycloid_pos(2 * np.arcsin(np.clip(np.sin(t0/2) * np.cos(omega * time_tracker.get_value()), -1, 1)))
            )

        for i, marble in enumerate(marbles):
            marble.add_updater(get_updater(theta_starts[i]))

        # Animate descent
        self.play(time_tracker.animate.set_value(T_descent), run_time=T_descent, rate_func=linear)
        
        # Keep them at the bottom for a moment
        self.wait(0.5)
        for marble in marbles:
            marble.clear_updaters()
            
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # "Descent time is independent of the starting position."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Highlight that they all reached the bottom simultaneously
        collision_circle = Circle(radius=0.3, color=WHITE).move_to(get_cycloid_pos(0))
        self.play(Create(collision_circle))
        self.play(FadeOut(collision_circle))
        
        self.wait(2)
