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
        # Setup layout
        title_text = "Case Study: Building a Square Wave"
        lecture_lines = [
            "Let's build a square wave using only odd harmonics.",
            "We start with a single fundamental sine wave.",
            "Adding the third harmonic begins sharpening the corners.",
            "Successive harmonics flatten the peaks and steepen the edges.",
            "Summing to infinity creates a perfect robotic square."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        WAVE_COLOR = WHITE
        OVERSHOOT_COLOR = "#FF4500" # Orange
        ACTIVE_COLOR = YELLOW
        ROBOT_COLOR = BLUE_C

        # Assets
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg]
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg", height=1.2)
        robot.set_color(ROBOT_COLOR)
        self.place_at_grid(robot, "A6") # Positioned near top right

        # Axes - Fixed per Issues 34 & 35
        axes = Axes(
            x_range=[0, 2 * PI, PI],
            y_range=[-1.5, 1.5, 1],
            x_length=5,
            y_length=3.5,
            axis_config={"include_tip": False, "color": GRAY}
        )
        self.place_in_area(axes, "B2", "F6", scale_factor=1.0)
        
        # Trackers
        n_tracker = ValueTracker(1) # Number of terms in the sum (1=sin(x), 2=sin(x)+sin(3x)/3...)
        t_tracker = ValueTracker(0) # Time for oscillation
        
        # Fourier sum function
        def fourier_sum_val(x, n_terms, t):
            total = 0
            n_count = int(n_terms)
            for i in range(1, n_count + 1):
                k = 2 * i - 1
                # Coefficient 1/k as per storyboard sin(3x)/3
                total += (1.0 / k) * np.sin(k * (x - t))
            return total

        # Wave mobject - optimized per Instruction 11
        wave = VMobject(color=WAVE_COLOR)
        wave.set_stroke(width=3)
        
        def update_wave(m):
            n = n_tracker.get_value()
            t = t_tracker.get_value()
            num_points = 300 # Sufficient for smooth square wave at n=20
            x_vals = np.linspace(0, 2 * PI, num_points)
            points = [axes.c2p(x, fourier_sum_val(x, n, t)) for x in x_vals]
            m.set_points_as_corners(points)

        wave.add_updater(update_wave)
        
        # Continuous oscillation updater for the tracker
        t_tracker.add_updater(lambda m, dt: m.increment_value(dt * 1.5))
        self.add(t_tracker)

        # === Animation for Lecture Line 1 ===
        # Let's build a square wave using only odd harmonics.
        self.lecture[0].set_color(ACTIVE_COLOR)
        self.play(Create(axes), FadeIn(robot, shift=LEFT))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # We start with a single fundamental sine wave.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(ACTIVE_COLOR)
        # Initial wave creation
        self.play(Create(wave))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Adding the third harmonic begins sharpening the corners.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(ACTIVE_COLOR)
        # Update n_tracker to include the 3rd harmonic (n=2 terms)
        self.play(n_tracker.animate.set_value(2), run_time=2)
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # Successive harmonics flatten the peaks and steepen the edges.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(OVERSHOOT_COLOR) # Matching color for highlights
        
        # Highlights for overshoot (Gibbs phenomenon)
        highlights = VGroup(
            Dot(color=OVERSHOOT_COLOR, radius=0.08),
            Dot(color=OVERSHOOT_COLOR, radius=0.08),
            Dot(color=OVERSHOOT_COLOR, radius=0.08),
            Dot(color=OVERSHOOT_COLOR, radius=0.08)
        )
        
        def update_highlights(m):
            t = t_tracker.get_value()
            n = n_tracker.get_value()
            n_harm = 2 * int(n) - 1
            # Heuristic for peak of Gibbs phenomenon overshoot near jumps
            delta = PI / (2 * n_harm)
            # Cycle through offsets corresponding to overshoot peaks
            offsets = [delta, PI - delta, PI + delta, 2 * PI - delta]
            for i, offset in enumerate(offsets):
                x_val = (offset + t) % (2 * PI)
                y_val = fourier_sum_val(x_val, n, t)
                m[i].move_to(axes.c2p(x_val, y_val))

        highlights.add_updater(update_highlights)
        
        # Add 5th and 7th harmonics (n=4 terms) and show highlights
        self.play(
            n_tracker.animate.set_value(4),
            FadeIn(highlights),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # Summing to infinity creates a perfect robotic square.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(ACTIVE_COLOR)
        
        # Robot "points" to result - a slight tilt/nudge
        # Increase terms to 20 and remove highlights
        self.play(
            n_tracker.animate.set_value(20),
            FadeOut(highlights),
            robot.animate.rotate(0.2, about_point=robot.get_center()).scale(1.1),
            run_time=5
        )
        self.wait(4)
        
        # Final reset
        self.lecture[4].set_color(WHITE)
        self.play(robot.animate.rotate(-0.2).scale(1/1.1))
        self.wait(2)
