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
        # Setup the layout with titles and lecture lines
        title_text = "The Euler Product: The Prime Connection"
        lecture_lines = [
            "- The zeta function transforms into a product over primes.",
            "- Animated gears represent primes building every single natural number.",
            "- This bridge links basic counting to the depth of primes."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors for lecture lines
        color_line1 = BLUE_A
        color_line2 = GREEN_A
        color_line3 = YELLOW_A

        # === Animation for Lecture Line 1 ===
        # The zeta function transforms into a product over primes.
        self.play(self.lecture[0].animate.set_color(color_line1))

        # Using MathTex for the Zeta function and Euler Product
        # If MathTex fails, we use Text as a fallback mechanism for robustness
        try:
            zeta_sum = MathTex(r"\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}", color=color_line1)
            euler_prod = MathTex(r"= \prod_{p \in \text{primes}} \frac{1}{1-p^{-s}}", color=color_line1)
        except:
            zeta_sum = Text("Zeta(s) = Sum 1/n^s", color=color_line1, font_size=24)
            euler_prod = Text("= Product 1/(1-p^-s)", color=color_line1, font_size=24)

        self.place_at_grid(zeta_sum, "B3", scale_factor=0.8)
        self.place_at_grid(euler_prod, "C3", scale_factor=0.8)

        self.play(FadeIn(zeta_sum))
        self.wait(1)
        self.play(FadeIn(euler_prod))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animated gears represent primes building every single natural number.
        self.play(self.lecture[1].animate.set_color(color_line2))

        # Prime Gears Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/ge.svg
        gear_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/ge.svg"
        
        gears = VGroup()
        prime_values = [2, 3, 5, 7]
        grid_positions = ["E2", "E3", "E4", "E5"]
        
        for p, pos in zip(prime_values, grid_positions):
            # Load SVG and add a label
            gear = SVGMobject(gear_path, color=color_line2, height=0.8)
            label = Text(str(p), font_size=18, color=WHITE)
            # Group gear and label
            gear_unit = VGroup(gear, label)
            self.place_at_grid(gear_unit, pos, scale_factor=0.9)
            gears.add(gear_unit)

        self.play(LaggedStart(*[FadeIn(g) for g in gears], lag_ratio=0.3))
        
        # Add updaters for rotation
        for i, gear_unit in enumerate(gears):
            # Alternate rotation direction for visual variety
            direction = 1 if i % 2 == 0 else -1
            gear_unit[0].add_updater(lambda m, dt, d=direction: m.rotate(d * dt * 0.5))

        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # This bridge links basic counting to the depth of primes.
        self.play(self.lecture[2].animate.set_color(color_line3))

        # Natural numbers flow out
        numbers = VGroup()
        for i in range(1, 11):
            num = Text(str(i), font_size=24, color=color_line3)
            self.place_at_grid(num, "E1", scale_factor=1.0) # Start from the left gear
            numbers.add(num)

        # Animation of numbers flowing
        flow_anims = []
        for i, num in enumerate(numbers):
            # Move from E1 to F6 area
            dest = self.grid["F6"] + RIGHT * (i * 0.4)
            flow_anims.append(num.animate(run_time=2, rate_func=linear).move_to(dest).set_opacity(0))

        self.play(LaggedStart(*flow_anims, lag_ratio=0.2))
        
        self.wait(2)

        # Clean up updaters
        for gear_unit in gears:
            gear_unit[0].clear_updaters()
        
        # Final transition
        self.play(
            FadeOut(zeta_sum),
            FadeOut(euler_prod),
            FadeOut(gears),
            FadeOut(self.title),
            FadeOut(self.lecture)
        )
