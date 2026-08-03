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
        # Lecture lines for Section 5
        lines = [
            "The cycloid also possesses the remarkable Tautochrone property.",
            "Objects released from any height reach the bottom simultaneously.",
            "Higher starting points cover more distance but move faster.",
            "These factors perfectly cancel out to keep time constant.",
            "This consistency confirms the cycloid's mathematical perfection."
        ]
        
        self.setup_layout("Properties and Verification", lines)

        # Helper for Cycloid point calculation
        # x = r(theta + sin(theta)), y = r(1 - cos(theta))
        # theta in [-PI, 0] for a slide starting high on left and ending low at 0
        r_val = 0.7
        def get_cycloid_point(theta):
            return np.array([r_val * (theta + np.sin(theta)), r_val * (1 - np.cos(theta)), 0])

        # Create Ramp
        cycloid_path = ParametricFunction(
            get_cycloid_point,
            t_range=[-PI, 0],
            color="#C0C0C0"
        )
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Fix Issue 36: Move ramp further right and scale up
        self.place_in_area(cycloid_path, "B3", "F6", scale_factor=1.4)
        self.play(Create(cycloid_path))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Define starting thetas for three balls
        thetas_start = [-0.9 * PI, -0.6 * PI, -0.3 * PI]
        colors = ["#FF00FF", "#00FF00", "#00FFFF"]
        balls = VGroup()
        markers = VGroup()
        for i, (col, theta) in enumerate(zip(colors, thetas_start)):
            ball = Dot(radius=0.08, color=col)
            # Map theta to proportion [0, 1] on the path [-PI, 0]
            pos = cycloid_path.point_from_proportion((theta + PI) / PI)
            ball.move_to(pos)
            # Create a small marker at the start position
            marker = Dot(radius=0.04, color=col).move_to(pos)
            balls.add(ball)
            markers.add(marker)

        self.play(FadeIn(markers), FadeIn(balls))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Time tracker for the harmonic motion
        time_tracker = ValueTracker(0)

        def update_ball(ball, theta_0):
            t_curr = time_tracker.get_value()
            # Exact cycloid tautochrone physics:
            # theta(t) = 2 * arcsin(sin(theta_0/2) * cos(pi/2 * t))
            cos_val = np.cos((PI / 2) * t_curr)
            sin_half_theta0 = np.sin(theta_0 / 2)
            current_theta = 2 * np.arcsin(np.clip(sin_half_theta0 * cos_val, -1, 1))
            
            # Map theta to proportion [0, 1] on the path [-PI, 0]
            prop = (current_theta + PI) / PI
            ball.move_to(cycloid_path.point_from_proportion(np.clip(prop, 0, 1)))

        # Add updaters
        for i in range(len(balls)):
            balls[i].add_updater(lambda b, idx=i: update_ball(b, thetas_start[idx]))

        # Start motion - release the balls
        # The tautochrone property says they reach the bottom simultaneously.
        self.play(time_tracker.animate.set_value(1), run_time=2.5, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # All balls collide at the bottom
        collision_point = cycloid_path.get_end()
        arrival_flash = Flash(collision_point, color=WHITE, flash_radius=0.5)
        
        # QED symbol as requested by storyboard
        qed_symbol = Text("QED", font_size=36, color="#FFD700")
        self.place_at_grid(qed_symbol, "F6", scale_factor=0.8)

        self.play(arrival_flash, FadeIn(qed_symbol))
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Clear updaters for static end
        for b in balls:
            b.clear_updaters()
            
        self.play(Indicate(qed_symbol, color="#FFD700", scale_factor=1.2))
        self.wait(3)
