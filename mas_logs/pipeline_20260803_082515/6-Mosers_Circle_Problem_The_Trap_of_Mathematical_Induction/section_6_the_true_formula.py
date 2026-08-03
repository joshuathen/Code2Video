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

class Section6TheTrueFormulaScene(TeachingScene):
    def construct(self):
        # Data
        title_text = "The Geometry of Regions: Combinatorics"
        lecture_lines = [
            "Why does the pattern fail at six points?",
            "The formula uses combinations of points and chords.",
            "Regions come from intersections and the lines themselves.",
            "This formula gives thirty-one for six points.",
            "Logic reveals what simple counting might miss."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Why does the pattern fail at six points?
        # Display R = binom(n,4) + binom(n,2) + 1 in cyan (#00FFFF).
        self.lecture[0].set_color("#00FFFF")
        
        formula = MathTex(
            "R", "=", "\\binom{n}{4}", "+", "\\binom{n}{2}", "+", "1",
            color="#00FFFF"
        )
        self.place_in_area(formula, "B2", "B5", scale_factor=1.2)
        
        self.play(Write(formula))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # The formula uses combinations of points and chords.
        # Highlight binom(n,2) and label it 'Chords' (#00FF00).
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FF00")
        
        chord_label = Text("Chords", font_size=20, color="#00FF00")
        # Resolution of Issue 39: Adjusting chord_label position and scale
        self.place_at_grid(chord_label, "C4", scale_factor=0.8)
        
        self.play(
            formula[4].animate.set_color("#00FF00"),
            FadeIn(chord_label, shift=UP)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Regions come from intersections and the lines themselves.
        # Highlight binom(n,4) and label it 'Intersection Points' (#FF0000).
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF0000")
        
        intersection_label = Text("Intersection Points", font_size=20, color="#FF0000")
        # Resolution of Issue 38: Adjusting intersection_label position and scale to avoid cramping
        self.place_in_area(intersection_label, "C2", "C3", scale_factor=0.8)
        
        self.play(
            formula[2].animate.set_color("#FF0000"),
            FadeIn(intersection_label, shift=UP)
        )
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # This formula gives thirty-one for six points.
        # Substitute n=6: 15 + 15 + 1 = 31 (#FFFFFF).
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FFFFFF")
        
        calc = MathTex("15", "+", "15", "+", "1", "=", "31", color="#FFFFFF")
        self.place_in_area(calc, "E2", "E5", scale_factor=1.2)
        
        self.play(FadeIn(calc, shift=DOWN))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # Logic reveals what simple counting might miss.
        # Highlight the final sum 31 in gold (#FFD700).
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFD700")
        
        highlight_box = SurroundingRectangle(calc[6], color="#FFD700", buff=0.1)
        
        self.play(
            calc[6].animate.set_color("#FFD700"),
            Create(highlight_box)
        )
        self.wait(3)
