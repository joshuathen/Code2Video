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
        # Setup title and lecture lines from storyboard
        self.setup_layout("Independence: The 'No Influence' Zone", [
            "Independence means event B provides no new information.",
            "Knowing B happened doesn't change the probability of A.",
            "The proportion of A remains constant across the universe."
        ])
        
        # Colors - Hex format
        COLOR_A = "#FFFF00"  # Yellow
        COLOR_B = "#00FF00"  # Green
        COLOR_INTERSECT = "#00FFFF"  # Cyan
        
        # === Animation for Lecture Line 1 ===
        # "Independence means event B provides no new information."
        self.play(self.lecture[0].animate.set_color(COLOR_A))
        
        # Universe Square - Positioning in C3-D5 to allow space for labels in B/E (B005)
        universe = Square(side_length=2.0, color=WHITE, stroke_width=2)
        self.place_in_area(universe, "C3", "D5")
        
        # Universe label in Row B (B005)
        u_label = Text("Universe", font_size=18, color=WHITE)
        self.place_at_grid(u_label, "B3", scale_factor=0.6)
        
        # Two independent circles A and B. They should overlap to show intersection.
        circle_a = Circle(radius=0.7, color=COLOR_A, fill_opacity=0.2)
        circle_b = Circle(radius=0.7, color=COLOR_B, fill_opacity=0.2)
        
        # Centers within the universe square area
        # Placing circle_a at C4 and circle_b at D4 creates vertical overlap
        self.place_at_grid(circle_a, "C4")
        self.place_at_grid(circle_b, "D4")
        
        label_a = Text("A", font_size=24, color=COLOR_A)
        label_b = Text("B", font_size=24, color=COLOR_B)
        
        # Labels fixed to be within 1 grid unit (Issue 38, 39)
        self.place_at_grid(label_a, "C3", scale_factor=1.0)
        self.place_at_grid(label_b, "D5", scale_factor=1.0)
        
        self.play(Create(universe), FadeIn(u_label))
        self.play(FadeIn(circle_a, label_a), FadeIn(circle_b, label_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Knowing B happened doesn't change the probability of A."
        self.play(self.lecture[1].animate.set_color(COLOR_B))
        
        # Asset Integration (Issue 31)
        # Highlight circle A first
        self.play(circle_a.animate.set_fill(opacity=0.5))
        
        # Highlight [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/the.svg] portion of A in B.
        # Place asset at the midpoint between C4 and D4
        the_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/the.svg")
        the_asset.set_color(COLOR_INTERSECT)
        self.place_in_area(the_asset, "C4", "D4", scale_factor=0.4)
        
        # Intersection area highlight
        intersect = Intersection(circle_a, circle_b, color=COLOR_INTERSECT, fill_opacity=0.8)
        
        self.play(FadeIn(intersect), FadeIn(the_asset))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The proportion of A remains constant across the universe."
        self.play(self.lecture[2].animate.set_color(COLOR_INTERSECT))
        
        # Proportion text in Row B (B005, avoid A)
        ratio_text = MathTex(
            "\\frac{P(A \\cap B)}{P(B)} = P(A)",
            font_size=32, color=COLOR_INTERSECT
        )
        self.place_in_area(ratio_text, "B4", "B6", scale_factor=0.6)
        
        # Formula in Row E (B005, avoid F)
        formula = MathTex("P(A|B) = P(A)", color=COLOR_INTERSECT)
        self.place_in_area(formula, "E4", "E6", scale_factor=0.6)
        
        self.play(FadeIn(ratio_text))
        self.play(Write(formula))
        self.wait(3)
