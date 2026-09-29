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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Functions define y clearly using x.",
            "Implicit relations trap y inside equations.",
            "We visualize these as geometric structures."
        ]
        self.setup_layout("Prerequisite Review: Explicit vs. Implicit", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        text_explicit = Text("Explicit", font_size=24, color="#FFFFFF")
        text_implicit = Text("Implicit", font_size=24, color="#FFFFFF")
        self.place_at_grid(text_explicit, 'B2')
        self.place_at_grid(text_implicit, 'B5')
        self.play(FadeIn(text_explicit), FadeIn(text_implicit))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        eq_explicit = MathTex("y = 2x + 1", color="#FF00FF")
        eq_implicit = MathTex("x^2 + y^2 = 1", color="#00FFFF")
        
        # Fixed layout per feedback
        self.place_in_area(eq_explicit, 'C2', 'C3', scale_factor=0.9)
        self.place_in_area(eq_implicit, 'C4', 'C5', scale_factor=0.9)
        self.play(Write(eq_explicit), Write(eq_implicit))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        formula_group = VGroup(eq_explicit, eq_implicit)
        self.place_in_area(formula_group, 'C2', 'C5', scale_factor=0.85)
        
        # Optional: Add lines/other visuals if needed
        self.wait(2)
