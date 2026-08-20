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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Synthesis & The Universal Framework", [
            "All these systems follow identical axioms.", 
            "Linear algebra applies to every domain.", 
            "Mastering the logic empowers every field."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show combination of axiom circle and vector list color #FFFFFF.
        axiom_circle = Circle(color=WHITE, radius=0.8)
        vector_list = VGroup(*[Dot(color=WHITE) for _ in range(5)]).arrange(DOWN)
        combo = VGroup(axiom_circle, vector_list).arrange(RIGHT, buff=0.5)
        # Applying Fix 31/43:
        self.place_in_area(combo, "E4", "F6", scale_factor=0.7)
        self.play(Create(combo))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Fade in "Linear Algebra" text overlay color #FF00FF.
        la_text = Text("Linear Algebra", font_size=36, color="#FF00FF")
        # Applying Fix 30/42:
        self.place_at_grid(la_text, "D4", scale_factor=0.8)
        self.play(FadeIn(la_text))
        self.lecture[1].set_color("#FF00FF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate merge of two structures into one unified framework color #FFFF00.
        unified = Rectangle(color="#FFFF00", width=2.5, height=1.5)
        unified_text = Text("Unified Framework", font_size=20, color="#FFFF00")
        unified_group = VGroup(unified, unified_text).arrange(DOWN)
        
        self.play(
            ReplacementTransform(combo, unified_group),
            ReplacementTransform(la_text, unified_group),
            run_time=2
        )
        # Applying Fix 29/41:
        self.place_in_area(unified_group, "A4", "C6", scale_factor=0.6)
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
