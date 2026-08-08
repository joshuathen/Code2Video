from manim import *

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
        # Data from storyboard and script
        title = "The 8 Axioms: The 'Membership Rules'"
        lines = [
            "A vector space follows eight strict rules.",
            "These axioms ensure consistent addition and scaling.",
            "Every space must contain a zero element.",
            "Every element must have an additive inverse.",
            "Passing these tests defines a vector space."
        ]
        
        self.setup_layout(title, lines)

        # Colors
        COLOR_BLUE = "#ADD8E6"
        COLOR_GREEN = "#90EE90"

        # Axiom expressions
        axiom_comm = MathTex("u + v = v + u", color=COLOR_BLUE)
        axiom_assoc = MathTex("(u + v) + w = u + (v + w)", color=COLOR_BLUE)
        axiom_ident = MathTex("v + 0 = v", color=COLOR_BLUE)
        axiom_inv = MathTex("v + (-v) = 0", color=COLOR_BLUE)
        axiom_dist = MathTex("a(u + v) = au + av", color=COLOR_BLUE)

        # === Animation for Lecture Line 1 ===
        # "A vector space follows eight strict rules."
        self.play(self.lecture[0].animate.set_color(COLOR_BLUE))
        # Issue 32 fix: Place axiom_comm at B4-B6 to avoid lecture overlap.
        self.place_in_area(axiom_comm, "B4", "B6", scale_factor=0.8)
        self.play(Write(axiom_comm))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "These axioms ensure consistent addition and scaling."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_BLUE)
        )
        # Issue 33 fix: Place axiom_assoc at D4-D6 to avoid lecture overlap.
        self.place_in_area(axiom_assoc, "D4", "D6", scale_factor=0.8)
        self.play(Write(axiom_assoc))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Every space must contain a zero element."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_BLUE)
        )
        # Place axiom_ident at C4-C6.
        self.place_in_area(axiom_ident, "C4", "C6", scale_factor=0.8)
        self.play(Write(axiom_ident))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Every element must have an additive inverse."
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_BLUE)
        )
        # Place axiom_inv at E4-E6.
        self.place_in_area(axiom_inv, "E4", "E6", scale_factor=0.8)
        self.play(Write(axiom_inv))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Passing these tests defines a vector space."
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(COLOR_GREEN)
        )
        
        # Issue 34 fix: Place axiom_dist at F4-F6 to avoid lecture overlap.
        self.place_in_area(axiom_dist, "F4", "F6", scale_factor=0.8)
        self.play(Write(axiom_dist))
        
        # Flash and turn green
        axioms_group = VGroup(axiom_comm, axiom_assoc, axiom_ident, axiom_inv, axiom_dist)
        self.play(
            Flash(axioms_group, color=COLOR_GREEN, flash_radius=2),
            axioms_group.animate.set_color(COLOR_GREEN)
        )
        self.wait(2)
