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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Inverses exist when space is preserved.", "No crushing ensures perfect reconstruction.", "These concepts unify linear systems theory."]
        self.setup_layout("Summary and Synthesis", lecture_lines)
        
        # Elements
        concept1 = Text("Inverse Exists", font_size=24, color=WHITE)
        concept2 = Text("Space Preserved", font_size=24, color=WHITE)
        concept3 = Text("No Crushing", font_size=24, color=WHITE)
        concept4 = Text("Reconstruction", font_size=24, color=WHITE)
        concepts = VGroup(concept1, concept2, concept3, concept4)
        
        formula = MathTex("A^{-1} Ax = x", color="#FFFF33")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00")
        self.place_in_area(concepts, 'D2', 'F5', scale_factor=0.8)
        self.play(FadeIn(concepts))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF5733")
        self.play(
            FadeOut(concepts),
            FadeIn(self.place_at_grid(formula, 'E3', scale_factor=1.0)),
            run_time=2
        )

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF33")
        self.play(formula.animate.scale(1.5).move_to(self.grid["D3"]))
        self.wait(1)
        self.play(FadeOut(self.title), FadeOut(self.lecture), FadeOut(formula))
