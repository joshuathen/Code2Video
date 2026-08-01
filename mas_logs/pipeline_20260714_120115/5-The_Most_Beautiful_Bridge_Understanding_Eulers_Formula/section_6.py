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
        # Data from storyboard
        title_text = "Summary: Elegance in Unity"
        lecture_lines = [
            "Consider five fundamental constants of the mathematical universe.",
            "They unite perfectly in this one simple, elegant statement.",
            "Zero, one, e, i, and pi combined forever.",
            "This formula is the golden key to mathematical harmony.",
            "We have reached the bridge: elegance in unity."
        ]
        
        # Initialize layout
        self.setup_layout(title_text, lecture_lines)
        
        # Colors for the constants and assets
        COLOR_E = "#88CA5E"
        COLOR_PI = "#FF9D00"
        COLOR_I = "#58C4DD"
        COLOR_1 = "#FFFFFF"
        COLOR_0 = "#FFFFFF"
        COLOR_KEY = "#FFD700"

        # Asset path for the golden key icon
        key_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/key.svg"

        # Initialize the parts of the constants for Animation 1
        char_e = Text("e", color=COLOR_E)
        char_i = Text("i", color=COLOR_I)
        char_pi = Text("π", color=COLOR_PI)
        char_one = Text("1", color=COLOR_1)
        char_zero = Text("0", color=COLOR_0)

        # Target formula parts for Animation 2 morphing
        # We construct the formula from individual Text objects to allow precise morphing and coloring.
        t_e = Text("e", color=COLOR_E)
        t_exp = Text("^(", color=WHITE)
        t_i = Text("i", color=COLOR_I)
        t_pi = Text("π", color=COLOR_PI)
        t_paren_r = Text(")", color=WHITE)
        t_plus = Text(" + ", color=WHITE)
        t_one = Text("1", color=COLOR_1)
        t_equals = Text(" = ", color=WHITE)
        t_zero = Text("0", color=COLOR_0)
        
        # Group formula parts and position it using place_in_area as per Issue 38
        # We don't add them yet; they serve as targets for ReplacementTransform.
        formula_group = VGroup(t_e, t_exp, t_i, t_pi, t_paren_r, t_plus, t_one, t_equals, t_zero).arrange(RIGHT, buff=0.1)
        # Fixed spatial balance issue as per Issue 38 instruction
        self.place_in_area(formula_group, 'C1', 'D6', scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        # Script: "Consider five fundamental constants of the mathematical universe."
        # Action: Arrange the symbols 0, 1, e, i, and pi randomly on the screen.
        self.lecture[0].set_color(YELLOW)
        
        # Place constants at random grid points on the right side
        self.place_at_grid(char_e, "B2", scale_factor=1.2)
        self.place_at_grid(char_i, "E4", scale_factor=1.2)
        self.place_at_grid(char_pi, "C5", scale_factor=1.2)
        self.place_at_grid(char_one, "B5", scale_factor=1.2)
        self.place_at_grid(char_zero, "E2", scale_factor=1.2)

        self.play(
            FadeIn(char_e), FadeIn(char_i), FadeIn(char_pi), FadeIn(char_one), FadeIn(char_zero),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Script: "They unite perfectly in this one simple, elegant statement."
        # Action: Morph the layout into the structured equation e^{i*pi} + 1 = 0.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        self.play(
            ReplacementTransform(char_e, t_e),
            ReplacementTransform(char_i, t_i),
            ReplacementTransform(char_pi, t_pi),
            ReplacementTransform(char_one, t_one),
            ReplacementTransform(char_zero, t_zero),
            FadeIn(t_exp), FadeIn(t_paren_r), FadeIn(t_plus), FadeIn(t_equals),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Script: "Zero, one, e, i, and pi combined forever."
        # Action: Flash each constant in its unique color.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Flashing each constant in parallel to emphasize their distinct identities
        self.play(
            Indicate(t_e, color=COLOR_E),
            Indicate(t_i, color=COLOR_I),
            Indicate(t_pi, color=COLOR_PI),
            Indicate(t_one, color=COLOR_1),
            Indicate(t_zero, color=COLOR_0),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Script: "This formula is the golden key to mathematical harmony."
        # Action: Transform the equation into a single glowing golden key icon.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Asset integration (Issue 24/44): Load and place the golden key SVG
        key_icon = SVGMobject(key_path, color=COLOR_KEY)
        # Centering the key icon in a prominent central area of the grid
        self.place_in_area(key_icon, "B2", "E5", scale_factor=1.2)
        
        # All currently visible parts of the formula
        everything_formula = VGroup(t_e, t_exp, t_i, t_pi, t_paren_r, t_plus, t_one, t_equals, t_zero)
        
        self.play(
            ReplacementTransform(everything_formula, key_icon),
            run_time=2
        )
        # Adding a visual pulse to simulate "glowing"
        self.play(Indicate(key_icon, color=YELLOW_A), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Script: "We have reached the bridge: elegance in unity."
        # Action: Fade in the title 'Elegance in Unity' below the golden key.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # Final concluding label
        unity_text = Text("Elegance in Unity", color=COLOR_KEY)
        # Position label in the bottom row (Row F) to keep it below the key icon
        self.place_in_area(unity_text, "F1", "F6", scale_factor=0.8)

        self.play(
            FadeIn(unity_text, shift=UP),
            run_time=2
        )
        self.wait(2)
