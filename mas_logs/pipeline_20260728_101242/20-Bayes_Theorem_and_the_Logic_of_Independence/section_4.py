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
        # Setup the layout with section title and lecture lines
        self.setup_layout("Deriving Bayes' Theorem: The Bridge", [
            "The intersection probability can be calculated two ways.",
            "It equals P(A given B) times P(B).",
            "It also equals P(B given A) times P(A).",
            "Equating these gives us the core of Bayes' theorem.",
            "We isolate P(A given B) to find the posterior."
        ])
        
        # Define colors as per requirements
        COLOR_A = "#00FF00"      # P(A) green
        COLOR_BA = "#0000FF"     # P(B|A) blue
        COLOR_AB = "#FF0000"     # P(A|B) red
        COLOR_B = "#FFFFFF"      # P(B) white
        COLOR_GOLD = "#FFD700"   # For the glowing box and bridge
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        # Initial intersection equation: P(A \cap B) = P(A|B)P(B)
        # Fix Issue 38: Shift eq1 to row A
        eq1 = MathTex("P(A \\cap B)", "=", "P(A|B)", "P(B)", color=WHITE)
        self.place_in_area(eq1, "A2", "A5", scale_factor=0.9)
        self.play(Write(eq1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        # Highlight the conditional expansion part P(A|B)P(B)
        rect1 = SurroundingRectangle(eq1[2:], color=YELLOW, buff=0.1)
        self.play(Create(rect1))
        self.wait(0.5)
        self.play(FadeOut(rect1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        # Second intersection equation: P(A \cap B) = P(B|A)P(A)
        # Fix Issue 38: Shift eq2 to row B
        eq2 = MathTex("P(A \\cap B)", "=", "P(B|A)", "P(A)", color=WHITE)
        self.place_in_area(eq2, "B2", "B5", scale_factor=0.9)
        self.play(Write(eq2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(COLOR_GOLD))
        
        # Issue 27: Load bridge asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg]
        # We use it as a visual metaphor for the connection between the two expansions.
        bridge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg", color=COLOR_GOLD, fill_opacity=0.4)
        self.place_in_area(bridge, "C2", "C5", scale_factor=1.2)
        
        # Equate the two right sides: P(A|B)P(B) = P(B|A)P(A)
        # Fix Issue 38: Shift eq3 to row C
        eq3 = MathTex("P(A|B)", "P(B)", "=", "P(B|A)", "P(A)", color=WHITE)
        self.place_in_area(eq3, "C2", "C5", scale_factor=0.9)
        
        # Show derivation from existing equations
        self.play(
            FadeIn(bridge),
            ReplacementTransform(eq1[2:].copy(), eq3[0:2]),
            ReplacementTransform(eq2[2:].copy(), eq3[3:5]),
            Write(eq3[2])
        )
        
        # Glowing box around the core equivalence
        box = SurroundingRectangle(eq3, color=COLOR_GOLD, buff=0.15)
        self.play(Create(box))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(COLOR_AB))
        
        # Final formula: P(A|B) = (P(B|A)P(A)) / P(B)
        # Fix Issue 39: Move final_eq to row D
        final_eq = MathTex("P(A|B)", "=", "{ P(B|A)", "P(A)", "\\over", "P(B) }", color=WHITE)
        
        # Apply colors to specific terms as per storyboard
        # Indexing: 0:P(A|B), 1:=, 2:P(B|A), 3:P(A), 4:bar, 5:P(B)
        final_eq[0].set_color(COLOR_AB) # Posterior
        final_eq[2].set_color(COLOR_BA) # Likelihood
        final_eq[3].set_color(COLOR_A)  # Prior
        
        self.place_in_area(final_eq, "D2", "D5", scale_factor=1.0)
        
        # Clean up derivations and show isolated formula
        self.play(
            FadeOut(box),
            FadeOut(bridge),
            FadeOut(eq1),
            FadeOut(eq2),
            ReplacementTransform(eq3, final_eq)
        )
        
        # Labels for the terms
        label_post = Text("Posterior", font_size=18, color=COLOR_AB)
        label_lik = Text("Likelihood", font_size=18, color=COLOR_BA)
        label_prior = Text("Prior", font_size=18, color=COLOR_A)
        
        # Fix Issue 39 & 40: Position labels in row E and scale to 0.7
        self.place_at_grid(label_post, "E2", scale_factor=0.7)
        self.place_at_grid(label_lik, "E4", scale_factor=0.7)
        self.place_at_grid(label_prior, "E5", scale_factor=0.7)
        
        self.play(
            FadeIn(label_post),
            FadeIn(label_lik),
            FadeIn(label_prior)
        )
        self.wait(3)

        # Final cleanup for the section
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
