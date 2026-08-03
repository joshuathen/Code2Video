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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "The Universal Ratio: Reynolds Number (Re)",
            [
                "Re is the ratio of inertia to viscosity.",
                "High velocity and scale overcome the fluid's stickiness.",
                "This constant dictates if a turbulent structure emerges."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Formula: Re = Inertia / Viscosity
        # Inertia in yellow (#FFFF00), Viscosity in green (#00FF00)
        re_tex = MathTex("Re", "=", "\\frac{\\text{Inertia}}{\\text{Viscosity}}", font_size=36)
        re_tex.set_color_by_tex("Inertia", "#FFFF00")
        re_tex.set_color_by_tex("Viscosity", "#00FF00")
        
        # Fix Issue 28: Scale factor 1.0
        self.place_in_area(re_tex, 'A2', 'B5', scale_factor=1.0)
        self.play(Write(re_tex))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )

        # Tug-of-War Visual
        # Yellow rope (Inertia) pulling against Green rope (Viscosity)
        
        knot_tracker = ValueTracker(0) # Center is 0
        
        yellow_side = Line(color="#FFFF00", stroke_width=8)
        green_side = Line(color="#00FF00", stroke_width=8)
        center_knot = Dot(color=WHITE, radius=0.15)
        
        inertia_label = Text("Inertia", font_size=20, color="#FFFF00")
        viscosity_label = Text("Viscosity", font_size=20, color="#00FF00")
        
        tug_group = VGroup(yellow_side, green_side, center_knot, inertia_label, viscosity_label)
        
        # Fix Issue 26: Use area C2-D6
        self.place_in_area(tug_group, 'C2', 'D6', scale_factor=0.8)
        
        # Define relative positions for updaters
        left_end = tug_group.get_left()
        right_end = tug_group.get_right()
        
        # Use updaters for movement (Persistent mobjects)
        center_knot.add_updater(lambda m: m.move_to(
            (left_end + right_end) / 2 + RIGHT * knot_tracker.get_value()
        ))
        yellow_side.add_updater(lambda m: m.set_points_as_corners([left_end, center_knot.get_center()]))
        green_side.add_updater(lambda m: m.set_points_as_corners([center_knot.get_center(), right_end]))
        inertia_label.add_updater(lambda m: m.next_to(yellow_side, UP, buff=0.2))
        viscosity_label.add_updater(lambda m: m.next_to(green_side, UP, buff=0.2))

        self.play(Create(VGroup(yellow_side, green_side, center_knot)), FadeIn(inertia_label, viscosity_label))
        self.wait(0.5)
        
        # Inertia "wins" - high velocity and scale
        self.play(
            knot_tracker.animate.set_value(1.2),
            inertia_label.animate.scale(1.2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )

        # Comparison: Whale (High Re) vs Bacteria (Low Re)
        # Issue 20: Use Assets
        whale_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/whale.svg")
        whale_asset.set_color(BLUE_D)
        whale_label = Text("Whale (High Re)", font_size=18).next_to(whale_asset, DOWN)
        
        # Turbulence particles for Whale
        # We'll use a fixed set of dots and update their positions to simulate flow
        particles = VGroup(*[Dot(radius=0.03, color=WHITE) for _ in range(20)])
        for p in particles:
            p.move_to(whale_asset.get_right() + RIGHT * np.random.rand() * 1.5 + UP * (np.random.rand()-0.5) * 1.0)
            
        whale_group = VGroup(whale_asset, whale_label, particles)
        
        # Fix Issue 27: Use area E2-F3
        self.place_in_area(whale_group, 'E2', 'F3', scale_factor=0.9)
        
        # Create Bacteria area
        bacteria_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bacteria.svg")
        bacteria_asset.set_color(WHITE)
        bacteria_label = Text("Bacteria (Low Re)", font_size=18).next_to(bacteria_asset, DOWN)
        honey_rect = Rectangle(width=2.5, height=1.8, fill_color=GOLD_E, fill_opacity=0.3, stroke_width=1, color=GOLD_E)
        bacteria_group = VGroup(honey_rect, bacteria_asset, bacteria_label)
        self.place_in_area(bacteria_group, 'E4', 'F6', scale_factor=0.9)

        # Flow animation for whale particles
        # Using simple shift in updater
        def update_particles(mob, dt):
            center_x = whale_asset.get_center()[0]
            right_bound = center_x + 1.5
            for p in mob:
                p.shift(RIGHT * 0.4 * dt)
                p.shift(UP * np.sin(self.renderer.time * 10 + p.get_x() * 5) * 0.02)
                if p.get_x() > right_bound:
                    p.set_x(center_x + 0.2)
                    p.set_y(whale_asset.get_center()[1] + (np.random.rand()-0.5) * 0.8)

        particles.add_updater(update_particles)

        self.play(FadeIn(whale_group), FadeIn(bacteria_group))
        self.wait(4)

        # Cleanup
        particles.remove_updater(update_particles)
        self.play(
            FadeOut(re_tex),
            FadeOut(VGroup(yellow_side, green_side, center_knot, inertia_label, viscosity_label)),
            FadeOut(whale_group),
            FadeOut(bacteria_group),
            self.lecture[2].animate.set_color(WHITE)
        )
        self.wait(1)
