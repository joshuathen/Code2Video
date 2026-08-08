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
        # Section title and lecture lines
        title = "Prerequisite: Conditional Probability & The Shrinking Universe"
        lecture_lines = [
            "Conditional probability looks at a specific subset of outcomes.",
            "If event B occurs, our entire universe shrinks.",
            "We only care about A where it overlaps B."
        ]
        self.setup_layout(title, lecture_lines)
        
        # Define colors for elements
        color_a = BLUE_D
        color_b = GREEN_D
        color_overlap = YELLOW
        color_universe = GRAY_E

        # === Animation for Lecture Line 1 ===
        # Step 1: Show Venn Diagram with circles A and B within the universe asset.
        self.lecture[0].set_color(color_a)
        
        # Universe Asset Integration (Issue 30)
        # Using Area B2 to D6 to clear E and F for formula (Issue 36 fix)
        universe_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/universe.svg")
        universe_asset.set_color(color_universe).set_opacity(0.3)
        self.place_in_area(universe_asset, "B2", "D6")
        
        # Label for universe - placed at A4 (midpoint of B2-D6 span) to follow B012/B001/B005
        universe_label = Text("Universe (Ω)", font_size=16, color=WHITE)
        self.place_at_grid(universe_label, "A4", scale_factor=0.8)
        
        # Circles A and B
        circle_a = Circle(radius=1.0, color=color_a, fill_opacity=0.3)
        circle_b = Circle(radius=1.0, color=color_b, fill_opacity=0.3)
        
        # Positioning them to overlap within B2-D6
        self.place_at_grid(circle_a, "C3")
        self.place_at_grid(circle_b, "C5")
        
        label_a = Text("A", font_size=20, color=color_a)
        label_b = Text("B", font_size=20, color=color_b)
        # Anchor labels 1 unit away (B012)
        self.place_at_grid(label_a, "B3", scale_factor=0.8)
        self.place_at_grid(label_b, "B5", scale_factor=0.8)
        
        self.play(
            FadeIn(universe_asset),
            FadeIn(universe_label),
            run_time=1
        )
        self.play(
            FadeIn(circle_a),
            FadeIn(circle_b),
            Write(label_a),
            Write(label_b),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Step 2: Fade out everything except circle B and A \cap B.
        # "Shrink the universe"
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(color_b)
        
        # Create a highlight version of Circle B as the new universe
        circle_b_highlight = circle_b.copy().set_stroke(width=6, color=color_overlap)
        
        self.play(
            universe_asset.animate.set_opacity(0.05),
            universe_label.animate.set_opacity(0.1),
            circle_a.animate.set_stroke(opacity=0.2).set_fill(opacity=0.1),
            label_a.animate.set_opacity(0.3),
            Create(circle_b_highlight),
            circle_b.animate.set_fill(opacity=0.5),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Step 3: Show formula P(A|B) = P(A \cap B) / P(B) next to Venn.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(color_overlap)
        
        # Intersection highlight
        intersection = Intersection(circle_a, circle_b, color=color_overlap, fill_opacity=0.8)
        
        # Formula - Issue 37: scale factor 1.0 to fit grid space
        formula = MathTex(
            r"P(A|B) = \frac{P(A \cap B)}{P(B)}",
            font_size=36,
            color=WHITE
        )
        # Coloring substrings to match visuals
        formula.set_color_by_tex("A|B", color_overlap)
        formula.set_color_by_tex(r"A \cap B", color_overlap)
        formula.set_color_by_tex("P(B)", color_b)
        
        self.place_in_area(formula, "E2", "F6", scale_factor=1.0)
        
        self.play(
            FadeIn(intersection),
            Write(formula),
            run_time=2
        )
        self.wait(3)
