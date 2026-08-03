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
        # Initializing data from storyboard and outline
        title = "Conclusion: The Hidden Harmony"
        lines = [
            "Colliding blocks hide a beautiful circular secret.",
            "Linear physics and circular geometry are deeply connected.",
            "Pi appears even where no physical circles exist."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Overlay the colliding blocks [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg] 
        # and the velocity circle, then merge them into Pi (#FF00FF).
        self.play(self.lecture[0].animate.set_color("#FF00FF"))

        # Load blocks asset (Issue 22)
        # Using two instances to satisfy the symmetric layout requirement (Issue 27)
        block1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg").set_color(BLUE)
        block2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg").set_color(GREEN)
        
        # Position blocks and circle symmetrically (Issue 27)
        self.place_at_grid(block1, 'B2', scale_factor=0.8)
        self.place_at_grid(block2, 'B5', scale_factor=0.8)

        circle = Circle(radius=0.7, color=WHITE, stroke_width=4)
        self.place_in_area(circle, 'E3', 'E4', scale_factor=0.8)

        # Define the target Pi symbol early for the merge
        pi_symbol = MathTex(r"\pi", color="#FF00FF").scale(3)
        self.place_in_area(pi_symbol, "C2", "D5")

        self.play(FadeIn(block1), FadeIn(block2), Create(circle))
        self.wait(1)

        # Move them to overlap in the center before merging
        self.play(
            block1.animate.move_to(self.grid["C3"]),
            block2.animate.move_to(self.grid["C4"]),
            circle.animate.move_to(self.grid["D3"]),
            run_time=2
        )

        self.play(
            ReplacementTransform(VGroup(block1, block2, circle), pi_symbol),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display the text 'Hidden Geometry' (#FFFFFF) and scale it up for emphasis.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFFFF")
        )

        hidden_text = Text("Hidden Geometry", font_size=36, color=WHITE)
        # Position correctly within area for horizontal centering (Issue 26)
        self.place_in_area(hidden_text, 'A2', 'A5', scale_factor=1.0)

        self.play(Write(hidden_text))
        self.play(hidden_text.animate.scale(1.4), run_time=1.5)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Fade all elements to black, leaving only a glowing magenta Pi symbol (#FF00FF).
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FF00FF")
        )

        # Create glow effect for Pi
        glow_outer = pi_symbol.copy().set_stroke(width=20, opacity=0.2).set_color("#FF00FF")
        glow_inner = pi_symbol.copy().set_stroke(width=10, opacity=0.4).set_color("#FF00FF")

        # Fade out everything except the Pi symbol and its new glow
        self.play(
            FadeOut(hidden_text),
            FadeOut(self.title),
            FadeOut(self.lecture),
            pi_symbol.animate.scale(1.5).move_to(ORIGIN), # Move to visual center for final focus
            run_time=2
        )

        # Apply glow to the centered Pi symbol
        glow_outer.move_to(pi_symbol.get_center()).scale(1.5)
        glow_inner.move_to(pi_symbol.get_center()).scale(1.5)

        self.play(
            FadeIn(glow_outer),
            FadeIn(glow_inner),
            run_time=1.5
        )
        self.wait(3)
