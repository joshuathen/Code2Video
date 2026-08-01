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

class Section3Scene(TeachingScene):
    def construct(self):
        # Topic: Visualizing the Number: 2^256
        lecture_lines = [
            "2 to the 256 is an unfathomable number.",
            "Imagine every grain of sand on all Earth's beaches.",
            "Now, think of every atom in our entire galaxy.",
            "This number is nearly atoms in the observable universe.",
            "It is a mathematical fortress of impossible scale."
        ]
        
        self.setup_layout("Visualizing the Number: 2^256", lecture_lines)

        # Colors
        SAND_COLOR = "#C2B280"
        GOLD_COLOR = "#FFD700"
        GREY_COLOR = "#D3D3D3"
        GALAXY_COLOR = "#1E90FF"

        # Assets
        SAND_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sand.svg"

        # === Animation for Lecture Line 1 ===
        # Line: 2 to the 256 is an unfathomable number.
        self.lecture[0].set_color(GOLD_COLOR)
        
        # A single grain of sand [Asset: sand.svg] (#C2B280) highlighted
        grain = SVGMobject(SAND_ASSET).set_color(SAND_COLOR)
        # Fix for Issue 39: scale_factor=1.0 for better visibility
        self.place_in_area(grain, "C3", "D4", scale_factor=1.0)
        
        # Glowing effect around the grain
        glow = Arc(radius=0.5, angle=TAU, color=WHITE).set_stroke(opacity=0.4, width=2)
        glow.move_to(grain)
        
        self.play(FadeIn(grain), Create(glow), run_time=1)
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Line: Imagine every grain of sand on all Earth's beaches.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(SAND_COLOR)
        
        # Rapid zoom out to sphere of all sand (optimized dot count)
        sand_sphere = Circle(radius=1.8, color=SAND_COLOR, fill_opacity=0.2).set_stroke(width=1)
        np.random.seed(42)
        sand_dots = VGroup(*[
            Dot(
                point=[np.random.uniform(-1.6, 1.6), np.random.uniform(-1.6, 1.6), 0],
                radius=0.015,
                color=SAND_COLOR,
                fill_opacity=np.random.uniform(0.4, 0.7)
            ) for _ in range(80) 
        ])
        
        # Keep dots inside circle
        for dot in sand_dots:
            if np.linalg.norm(dot.get_center()) > 1.6:
                dot.set_opacity(0)
        
        sand_visual = VGroup(sand_sphere, sand_dots)
        self.place_in_area(sand_visual, "B2", "E5")
        
        self.play(
            FadeOut(glow),
            ReplacementTransform(grain, sand_visual),
            run_time=2,
            rate_func=smooth
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Line: Now, think of every atom in our entire galaxy.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GALAXY_COLOR)
        
        # Transform into a swirling galaxy (optimized star count)
        galaxy = VGroup()
        num_arms = 3
        stars_per_arm = 30
        for i in range(num_arms):
            start_angle = i * (TAU / num_arms)
            for j in range(stars_per_arm):
                r = (j / stars_per_arm) * 1.7 + 0.2
                theta = r * 3.0 + start_angle
                star = Dot(
                    point=[r * np.cos(theta), r * np.sin(theta), 0],
                    radius=0.012,
                    color=WHITE,
                    fill_opacity=np.random.uniform(0.5, 1.0)
                )
                galaxy.add(star)
        
        galaxy_center = Dot(radius=0.1, color=WHITE).set_style(fill_opacity=0.8)
        galaxy.add(galaxy_center)
        self.place_in_area(galaxy, "B2", "E5")
        
        # Persistent rotation updater for the galaxy
        galaxy.add_updater(lambda m, dt: m.rotate(0.1 * dt))

        self.play(
            ReplacementTransform(sand_visual, galaxy),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # Line: This number is nearly atoms in the observable universe.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(GREY_COLOR)
        
        # '2^256' in large gold text
        num_2_256 = MathTex("2^{256}", color=GOLD_COLOR, font_size=48)
        # Fix for Issue 37: Area positioning to avoid clutter and overlap
        self.place_in_area(num_2_256, 'A2', 'A5', scale_factor=1.0)
        
        # Comparison text
        comparison_text = Text("Nearly Total Atoms\nin Universe", color=GREY_COLOR, font_size=18)
        # Fix for Issue 38: Area positioning to avoid edge cutoff
        self.place_in_area(comparison_text, 'F2', 'F5', scale_factor=0.8)
        
        self.play(
            Write(num_2_256),
            FadeIn(comparison_text, shift=UP),
            run_time=1.5
        )
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        # Line: It is a mathematical fortress of impossible scale.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(GOLD_COLOR)
        
        # Final focus on the scale
        self.play(
            num_2_256.animate.scale(1.2),
            galaxy.animate.scale(0.8),
            run_time=2
        )
        self.wait(2)
        
        # Cleanup updaters
        galaxy.clear_updaters()
