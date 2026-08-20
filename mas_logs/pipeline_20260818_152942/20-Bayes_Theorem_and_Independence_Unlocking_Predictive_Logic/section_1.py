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
        self.setup_layout("Prerequisite Warm-up: Conditional Probability", [
            "Conditional probability defines likelihood given known info.",
            "P(A|B) is probability of A given B occurred.",
            "Sample space shrinks to region of B."
        ])
        
        # Elements
        formula = MathTex(r"P(A|B) = \frac{P(A \cap B)}{P(B)}", font_size=32, color=WHITE)
        
        # Venn Diagram representation
        universe = Circle(radius=1.5, color=WHITE, stroke_width=2)
        event_b = Circle(radius=0.8, color="#FF0000", fill_opacity=0.3)
        event_a_overlap = Intersection(Circle(radius=1.0), Circle(radius=0.8).shift(LEFT*0.5), color="#00FF00", fill_opacity=0.5)
        
        # Positioning based on feedback (Line numbers from prompt instructions)
        self.place_in_area(formula, 'B4', 'C6', scale_factor=0.6)
        self.place_in_area(universe, 'B3', 'D5', scale_factor=0.9)
        self.place_at_grid(event_b, 'C4', scale_factor=0.8)
        self.place_at_grid(event_a_overlap, 'C3', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(formula), run_time=1.5)
        self.play(self.lecture[0].animate.set_color("#FFD700"), run_time=0.5)

        # === Animation for Lecture Line 2 ===
        self.play(Create(universe), Create(event_b), run_time=2)
        self.play(self.lecture[1].animate.set_color("#FF0000"), run_time=0.5)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(event_a_overlap), run_time=1.5)
        self.play(self.lecture[2].animate.set_color("#00FF00"), run_time=0.5)
        self.play(Indicate(formula), run_time=1)
