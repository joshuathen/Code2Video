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
        title = "Visualizing the Scale: The Universal Perspective"
        lecture_lines = [
            "Imagine every grain of sand on Earth is a hash.",
            "All Earth's sand is only ten to the nineteenth grains.",
            "To reach our number, we need billions of Earths.",
            "The search space is wider than the observable universe.",
            "Finding one hash is impossible in this vast void."
        ]
        self.setup_layout(title, lecture_lines)
        
        # Assets
        SAND_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sand.svg"
        EARTH_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/earth.svg"
        
        # Colors
        SAND_COLOR = "#C2B280"
        EARTH_COLOR = "#1E90FF"
        KEY_COLOR = "#FFD700"
        ATOM_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Imagine every grain of sand on Earth is a hash.
        # Show a single grain of sand [Asset: sand.svg] (#C2B280) zooming out to a full beach.
        self.lecture[0].set_color(SAND_COLOR)
        grain = SVGMobject(SAND_ASSET).set_color(SAND_COLOR)
        self.place_at_grid(grain, 'C3', scale_factor=0.2)
        
        # A visual area for a "beach"
        beach_area = Rectangle(width=3, height=2, color=SAND_COLOR, fill_opacity=0.3, stroke_width=0)
        self.place_in_area(beach_area, 'C3', 'E5')
        
        # Scattered grains for the beach (using dots for performance stability)
        dots = VGroup(*[
            Dot(radius=0.02, color=SAND_COLOR).move_to(
                beach_area.get_center() + np.array([
                    np.random.uniform(-1.4, 1.4),
                    np.random.uniform(-0.9, 0.9),
                    0
                ])
            ) for _ in range(60)
        ])
        
        self.play(FadeIn(grain))
        self.wait(1)
        self.play(
            grain.animate.scale(0.5).move_to(beach_area.get_center()),
            FadeIn(beach_area),
            FadeIn(dots, lag_ratio=0.05),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # All Earth's sand is only ten to the nineteenth grains.
        # Zoom out to show Earth [Asset: earth.svg] (#1E90FF)
        self.lecture[1].set_color(EARTH_COLOR)
        earth = SVGMobject(EARTH_ASSET).set_color(EARTH_COLOR).set_fill(EARTH_COLOR, opacity=0.8)
        
        # Fix Label: Position 'earth_label' in area 'B3'-'B4' (Issue 43)
        earth_label = Text("10^19 grains", font_size=22, color=EARTH_COLOR)
        self.place_in_area(earth_label, 'B3', 'B4', scale_factor=0.8)
        
        self.place_at_grid(earth, 'D4', scale_factor=0.8)

        self.play(
            FadeOut(beach_area),
            FadeOut(dots),
            FadeOut(grain),
            FadeIn(earth),
            Write(earth_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # To reach our number, we need billions of Earths.
        # Multiply into a grid of planets [Asset: earth.svg].
        self.lecture[2].set_color(EARTH_COLOR)
        
        planets = VGroup()
        # Area B3 to E6 for the grid to keep it away from lecture text buffer
        for r in ["B", "C", "D", "E"]:
            for c in ["3", "4", "5", "6"]:
                p = SVGMobject(EARTH_ASSET).set_color(EARTH_COLOR).set_fill(EARTH_COLOR, opacity=0.8)
                self.place_at_grid(p, f"{r}{c}", scale_factor=0.25)
                planets.add(p)
        
        self.play(
            FadeOut(earth),
            FadeOut(earth_label),
            LaggedStart(*[FadeIn(p) for p in planets], lag_ratio=0.05),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The search space is wider than the observable universe.
        # Fade out planets and fill the screen with a dense cloud of white atom dots.
        self.lecture[3].set_color(ATOM_COLOR)
        
        # Dense cloud of atoms. 
        atom_cloud = VGroup(*[
            Dot(radius=0.015, color=ATOM_COLOR, fill_opacity=0.6).move_to(
                self.grid['D4'] + np.array([
                    np.random.uniform(-2.5, 2.5),
                    np.random.uniform(-2.5, 2.5),
                    0
                ])
            ) for _ in range(120)
        ])
        
        self.play(
            FadeOut(planets),
            FadeIn(atom_cloud, lag_ratio=0.01),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Finding one hash is impossible in this vast void.
        # Fix Overlap: Move 'key' to 'E6' (Issue 43)
        self.lecture[4].set_color(KEY_COLOR)
        
        # Golden key representation
        key = Star(n=5, color=KEY_COLOR, fill_opacity=1)
        self.place_at_grid(key, 'E6', scale_factor=0.15)
        
        # Fix Obstruction: Move 'spotlight' to 'C5' (Issue 43)
        spotlight = Circle(radius=0.9, color=WHITE, stroke_width=2)
        spotlight.set_fill(WHITE, opacity=0.2)
        self.place_at_grid(spotlight, 'C5')

        self.play(FadeIn(key))
        self.wait(0.5)
        self.play(Indicate(key, color=KEY_COLOR)) 
        
        # Key gets lost in the cloud
        self.play(key.animate.set_fill_opacity(0.1)) 
        
        self.play(FadeIn(spotlight))
        # Search animation
        self.play(
            spotlight.animate.move_to(self.grid['B6']),
            run_time=1.5,
            rate_func=linear
        )
        self.play(
            spotlight.animate.move_to(self.grid['F3']),
            run_time=2,
            rate_func=linear
        )
        self.play(FadeOut(spotlight))
        self.wait(2)
