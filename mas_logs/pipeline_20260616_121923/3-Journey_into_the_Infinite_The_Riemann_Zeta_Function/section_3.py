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
        # Setup layout with lecture lines
        lecture_lines = [
            "Euler linked the Zeta function to prime numbers.",
            "The sum transforms into an infinite product of primes.",
            "Zeta acts as a DNA map for all primes."
        ]
        self.setup_layout("The Prime Connection: The Euler Product Formula", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.lecture[0].set_color(WHITE)
        
        # Formulas (Sum and Product)
        # Using unicode Σ (U+03A3) and Π (U+03A0)
        zeta_sum_text = Text("ζ(s) = Σ 1/n^s", font_size=32, color=WHITE)
        zeta_prod_text = Text("= Π 1/(1-p^-s)", font_size=32, color=WHITE)
        zeta_formula = VGroup(zeta_sum_text, zeta_prod_text).arrange(RIGHT, buff=0.5)
        
        # Issue 40: Positioning zeta_formula
        self.place_in_area(zeta_formula, 'A2', 'B5', scale_factor=0.8)
        
        # Issue 41: Complex Plane
        complex_plane = ComplexPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": False, "stroke_width": 2}
        )
        self.place_in_area(complex_plane, 'C1', 'F6', scale_factor=0.75)
        
        # Issue 42: Axis labels
        axis_labels = VGroup(
            Text("1", font_size=16).next_to(complex_plane.n2p(1), DOWN, buff=0.1),
            Text("i", font_size=16).next_to(complex_plane.n2p(1j), LEFT, buff=0.1)
        )
        self.place_at_grid(axis_labels, 'D4', scale_factor=0.6)
        
        self.play(Write(zeta_formula), Create(complex_plane), FadeIn(axis_labels))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.play(self.lecture[0].animate.set_color(GRAY), self.lecture[1].animate.set_color("#7FFF00"))
        
        # Issue 32: Asset Integration - Gears
        # Lime Green: #7FFF00
        gears_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/gears.svg"
        
        gear1 = SVGMobject(gears_path).set_color("#7FFF00")
        gear2 = SVGMobject(gears_path).set_color("#7FFF00")
        gear3 = SVGMobject(gears_path).set_color("#7FFF00")
        
        # Labels for gears
        label2 = Text("2", font_size=20, color=WHITE).move_to(gear1.get_center())
        label3 = Text("3", font_size=20, color=WHITE).move_to(gear2.get_center())
        label5 = Text("5", font_size=20, color=WHITE).move_to(gear3.get_center())
        
        gear_group1 = VGroup(gear1, label2)
        gear_group2 = VGroup(gear2, label3)
        gear_group3 = VGroup(gear3, label5)
        
        # Position gears interlocking in the center
        self.place_at_grid(gear_group1, "D2", scale_factor=0.4)
        self.place_at_grid(gear_group2, "D3", scale_factor=0.4)
        self.place_at_grid(gear_group3, "D4", scale_factor=0.4)
        
        # Add rotation updaters
        gear1.add_updater(lambda m, dt: m.rotate(dt * 0.8))
        gear2.add_updater(lambda m, dt: m.rotate(-dt * 0.8))
        gear3.add_updater(lambda m, dt: m.rotate(dt * 0.8))
        
        # Ensure labels stay at centers (since they are in VGroups but the gear is rotating internally)
        label2.add_updater(lambda m: m.move_to(gear1.get_center()))
        label3.add_updater(lambda m: m.move_to(gear2.get_center()))
        label5.add_updater(lambda m: m.move_to(gear3.get_center()))

        self.play(
            FadeIn(gear_group1),
            FadeIn(gear_group2),
            FadeIn(gear_group3)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight third line
        self.play(self.lecture[1].animate.set_color(GRAY), self.lecture[2].animate.set_color("#FFD700"))
        
        # Transform gears into a golden bridge
        bridge_line = RoundedRectangle(corner_radius=0.2, height=0.4, width=3.5, color="#FFD700", fill_opacity=0.8)
        self.place_in_area(bridge_line, "C2", "D5", scale_factor=1.0)
        
        dna_text = Text("Primes = DNA", font_size=24, color="#FFD700")
        self.place_at_grid(dna_text, "C3", scale_factor=1.0)
        dna_text.shift(UP * 0.5) # Positioning relative to bridge
        
        # Stop updaters before transform
        gear1.clear_updaters()
        gear2.clear_updaters()
        gear3.clear_updaters()
        label2.clear_updaters()
        label3.clear_updaters()
        label5.clear_updaters()
        
        self.play(
            ReplacementTransform(VGroup(gear_group1, gear_group2, gear_group3), bridge_line),
            FadeIn(dna_text)
        )
        self.wait(3)
