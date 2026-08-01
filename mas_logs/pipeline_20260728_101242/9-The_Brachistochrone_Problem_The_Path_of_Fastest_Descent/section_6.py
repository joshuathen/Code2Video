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
        # Lecture lines for Section 6
        lecture_lines = [
            "The cycloid also has the unique tautochrone property.",
            "Objects released from anywhere reach the bottom simultaneously.",
            "This \"equal time\" property defines this remarkable curve."
        ]
        
        # Setup layout
        self.setup_layout("The Tautochrone Property & Application", lecture_lines)
        
        # Define the cycloid function for a bowl
        # Standard tautochrone: x = a(phi + sin(phi)), y = a(1 - cos(phi))
        # phi from -pi to pi. phi=0 is the bottom.
        a_val = 0.6
        def cycloid_func(phi):
            return np.array([
                a_val * (phi + np.sin(phi)),
                a_val * (1 - np.cos(phi)),
                0
            ])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE_B)
        
        # Create cycloid curve
        cycloid_curve = ParametricFunction(
            cycloid_func,
            t_range=[-np.pi, np.pi],
            color=BLUE_B
        )
        
        # Load and set up sphere assets (Issue 22)
        sphere_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg"
        sphere1 = SVGMobject(sphere_path).set_color(WHITE).set_height(0.25)
        sphere2 = SVGMobject(sphere_path).set_color(WHITE).set_height(0.25)
        
        # Initial positions (proportion logic)
        # P(phi) = (sin(phi/2) + 1) / 2
        phi1_start = -0.85 * np.pi
        phi2_start = -0.35 * np.pi
        
        prop1_start = (np.sin(phi1_start / 2) + 1) / 2
        prop2_start = (np.sin(phi2_start / 2) + 1) / 2
        
        sphere1.move_to(cycloid_curve.point_from_proportion(prop1_start))
        sphere2.move_to(cycloid_curve.point_from_proportion(prop2_start))
        
        # Create group and place according to VideoCritic (Issue 31)
        tautochrone_group = VGroup(cycloid_curve, sphere1, sphere2)
        self.place_in_area(tautochrone_group, 'A2', 'D6', scale_factor=0.9)
        
        # Note: After place_in_area, cycloid_curve's points are transformed.
        # point_from_proportion will correctly return global coordinates.
        
        self.play(Create(cycloid_curve), run_time=2)
        self.play(FadeIn(sphere1), FadeIn(sphere2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW_A)
        
        # ValueTracker for motion progress p (0 to 1)
        progress = ValueTracker(0)
        
        # Updaters using the tautochrone physics logic:
        # The equation of motion s'' = -g/4a * s leads to s(t) = s0 * cos(omega * t)
        # s = 4a * sin(phi/2)
        # so sin(phi(t)/2) = sin(phi0/2) * cos(pi/2 * p)
        # and prop(t) = (sin(phi(t)/2) + 1) / 2
        
        def update_sphere(m, start_phi):
            p = progress.get_value()
            current_sin_half_phi = np.sin(start_phi / 2) * np.cos(np.pi / 2 * p)
            current_prop = (current_sin_half_phi + 1) / 2
            m.move_to(cycloid_curve.point_from_proportion(current_prop))
            
        sphere1.add_updater(lambda m: update_sphere(m, phi1_start))
        sphere2.add_updater(lambda m: update_sphere(m, phi2_start))
        
        # Run release animation - they reach the bottom (prop=0.5) when p=1
        self.play(progress.animate.set_value(1), run_time=4, rate_func=linear)
        self.wait(1)
        
        sphere1.clear_updaters()
        sphere2.clear_updaters()

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN_A)
        
        # Load clock asset (Issue 22)
        clock_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/clock.svg"
        clock = SVGMobject(clock_path).set_color(YELLOW_A).set_height(0.6)
        
        # Highlight arrival at the bottom (phi=0, prop=0.5)
        arrival_point = cycloid_curve.point_from_proportion(0.5)
        
        # Position clock below the bottom point using grid alignment (Issue 31)
        self.place_at_grid(clock, 'E4', scale_factor=0.8)
        clock.move_to(arrival_point + DOWN * 0.8)
        
        success_flash = Flash(arrival_point, color=GREEN_A, flash_radius=0.5)
        
        self.play(FadeIn(clock), success_flash)
        self.wait(2)
