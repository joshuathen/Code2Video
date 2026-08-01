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

class Section7Scene(TeachingScene):
    def construct(self):
        # Setup
        lecture_lines = [
            'Prime distribution is vital for modern digital security.',
            'Zeta also describes energy levels in quantum physics.',
            'This bridge connects pure math to the physical world.'
        ]
        self.setup_layout("Summary and Real-World Impact", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))

        # Axes for prime counting
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 5, 1],
            x_length=4,
            y_length=3,
            axis_config={"color": WHITE, "include_tip": False}
        )
        self.place_in_area(axes, "B1", "E6", scale_factor=0.8)
        
        # Prime counting step function pi(x)
        primes = [2, 3, 5, 7]
        step_points = [[0, 0]]
        current_y = 0
        for p in primes:
            step_points.append([p, current_y])
            current_y += 1
            step_points.append([p, current_y])
        step_points.append([10, current_y])
        
        step_func = axes.plot_line_graph(
            x_values=[p[0] for p in step_points],
            y_values=[p[1] for p in step_points],
            line_color="#00FFFF",
            add_vertex_dots=False,
            stroke_width=4
        )
        
        # Smooth approximation curve (Yellow)
        smooth_curve = axes.plot(lambda x: x / np.log(x + 1.1) if x > 0 else 0, x_range=[1.1, 10], color="#FFFF00")

        self.play(Create(axes), Create(step_func), run_time=1.5)
        self.play(Create(smooth_curve), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            FadeOut(axes), FadeOut(step_func), FadeOut(smooth_curve),
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FF00")
        )

        # Load Padlock Asset (Issue 37)
        padlock = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/padlock.svg")
        padlock.set_color(WHITE)
        # Position Padlock (Issue 55)
        self.place_at_grid(padlock, "E3", scale_factor=1.0)

        # Euler Product Formula Text
        euler_formula = Text(
            "ζ(s) = ∏ₚ 1 / (1 - p⁻ˢ)",
            color=WHITE
        )
        # Position Euler Formula (Issue 54)
        self.place_in_area(euler_formula, "B2", "B5", scale_factor=0.9)

        self.play(FadeIn(padlock), FadeIn(euler_formula))
        self.wait(0.5)

        # Glow and Unlock animation
        # Open padlock by rotating and shifting (simulating opening since it's a single SVG group)
        self.play(
            euler_formula.animate.set_color("#00FF00"),
            padlock.animate.rotate(PI/12).shift(UP * 0.2),
            run_time=1.5
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        self.play(
            FadeOut(padlock), FadeOut(euler_formula),
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(WHITE)
        )

        # Zeta symbol
        zeta_symbol = Text("ζ(s)", color=WHITE, font_size=120)
        # Position Zeta Symbol (Issue 56)
        self.place_in_area(zeta_symbol, "B2", "E5", scale_factor=1.0)

        # Load Stars Asset (Issue 37)
        # We'll create a few instances of the star SVG for a twinkling effect
        star_positions = [
            self.grid["A2"], self.grid["A5"], self.grid["B6"],
            self.grid["C1"], self.grid["D5"], self.grid["E2"],
            self.grid["F4"], self.grid["F6"]
        ]
        
        stars = VGroup()
        for pos in star_positions:
            star = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/stars.svg")
            star.scale(0.15).move_to(pos + np.random.uniform(-0.3, 0.3, 3))
            star.set_color(WHITE)
            stars.add(star)
        
        # Twinkling effect using updaters
        def update_stars(m, dt):
            for star in m:
                # Randomly change opacity to simulate twinkling
                if np.random.rand() > 0.8:
                    star.set_fill(opacity=np.random.uniform(0.3, 1.0))

        self.add(stars)
        stars.add_updater(update_stars)

        self.play(
            FadeIn(zeta_symbol),
            zeta_symbol.animate.scale(1.5),
            run_time=4,
            rate_func=smooth
        )
        self.wait(2)
        
        # Cleanup updaters
        stars.remove_updater(update_stars)
        self.wait(1)
