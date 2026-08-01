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
        # Setup layout
        lecture_lines = [
            'Zeta reveals a deep connection to prime numbers.', 
            'It can be written as a product of all primes.', 
            'This formula encodes the distribution of primes like DNA.'
        ]
        self.setup_layout("The Golden Link: Primes and the Euler Product", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Show prime numbers as floating musical notes (#00FFFF).
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        
        primes = [2, 3, 5, 7, 11, 13]
        grid_positions = ["B2", "C5", "D1", "E4", "B6", "F2"]
        note_group = VGroup()
        
        for p, pos in zip(primes, grid_positions):
            # Create a simple musical note shape
            head = Ellipse(width=0.3, height=0.2, fill_opacity=1, color="#00FFFF", stroke_width=0)
            head.rotate(PI/6)
            stem = Line(start=head.get_right(), end=head.get_right() + UP*0.5, color="#00FFFF", stroke_width=3)
            label = Text(str(p), font_size=20, color=WHITE).next_to(head, DOWN, buff=0.1)
            note = VGroup(head, stem, label)
            self.place_at_grid(note, pos, scale_factor=0.8)
            note_group.add(note)
            
        self.play(LaggedStart(*[FadeIn(n, shift=UP*0.3) for n in note_group], lag_ratio=0.2))
        
        # Floating effect for notes
        for note in note_group:
            note.add_updater(lambda m, dt: m.shift(np.sin(self.renderer.time * 2) * 0.002 * UP))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Use Text instead of MathTex to avoid FileNotFoundError: 'latex'
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        # Formula: zeta(s) = Product_p (1 / (1 - p^-s))
        euler_formula = Text(
            "ζ(s) = ∏ [1 / (1 - p⁻ˢ)]",
            color="#FFFF00", font_size=28
        )
        # Position formula in the upper-middle grid area (Issue 49: B1-C6)
        self.place_in_area(euler_formula, 'B1', 'C6', scale_factor=1.0)
        
        self.play(Write(euler_formula))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # A green DNA helix (#00FF00) forms from the primes.
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        # Fade out notes to clear workspace (Issue 51)
        # Remove updaters before FadeOut to ensure clean transition
        for note in note_group:
            note.clear_updaters()
        self.play(FadeOut(note_group))
        
        # Load DNA helix asset (Issue 36)
        dna_helix = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/dna.svg")
        dna_helix.set_color("#00FF00")
        # Position DNA in the lower grid area (Issue 50: E1-F6)
        self.place_in_area(dna_helix, 'E1', 'F6', scale_factor=0.8)
        
        # Transform the product formula into the DNA helix (Issue 36)
        self.play(ReplacementTransform(euler_formula, dna_helix), run_time=3)
        
        # Final subtle pulse for DNA
        self.play(dna_helix.animate.scale(1.1), run_time=1)
        self.play(dna_helix.animate.scale(1/1.1), run_time=1)
        
        self.wait(3)
