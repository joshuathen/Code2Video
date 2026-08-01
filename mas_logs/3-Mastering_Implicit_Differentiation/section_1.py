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
        # 1. Setup layout with section-specific title and lecture lines
        self.setup_layout("The Hook: Explicit vs. Implicit", [
            "Meet explicit functions: y is isolated and clear.",
            "They are like simple recipes for y.",
            "Now meet implicit relations: x and y are tangled.",
            "This circle's equation locks variables together tightly.",
            "How do we find the slope without isolation?"
        ])

        # Define Colors from animation requirements
        BLUE_EXPLICIT = "#52ADFF"
        YELLOW_IMPLICIT = "#F7D038"

        # === Animation for Lecture Line 1 ===
        # "Meet explicit functions: y is isolated and clear."
        # Stage 1: Display 'Explicit: y = x² + 1' in Blue (#52ADFF) at the top of the screen.
        self.play(self.lecture[0].animate.set_color(BLUE_EXPLICIT))
        
        explicit_label = Text("Explicit: ", font_size=24, color=BLUE_EXPLICIT)
        explicit_val = Text("y = x^2 + 1", font_size=24, color=BLUE_EXPLICIT)
        explicit_group = VGroup(explicit_label, explicit_val).arrange(RIGHT)
        self.place_in_area(explicit_group, "A2", "A5")
        
        self.play(Write(explicit_group))
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # "They are like simple recipes for y."
        # Stage 3: Animate 'y = x² + 1' transforming into 'dy/dx = 2x' to show a clear path.
        self.play(self.lecture[1].animate.set_color(BLUE_EXPLICIT))
        
        explicit_deriv_val = MathTex("\\frac{dy}{dx} = 2x", color=BLUE_EXPLICIT)
        explicit_deriv_group = VGroup(explicit_label.copy(), explicit_deriv_val).arrange(RIGHT)
        self.place_in_area(explicit_deriv_group, "A2", "A5")
        
        self.play(Transform(explicit_group, explicit_deriv_group))
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # "Now meet implicit relations: x and y are tangled."
        # Stage 2: Display 'Implicit: x² + y² = 25' in Yellow (#F7D038) at the bottom.
        self.play(self.lecture[2].animate.set_color(YELLOW_IMPLICIT))
        
        implicit_label = Text("Implicit: ", font_size=24, color=YELLOW_IMPLICIT)
        # Split MathTex for highlighting specific variables later
        implicit_val = MathTex("x^2", "+", "y^2", "=", "25", color=YELLOW_IMPLICIT)
        implicit_group = VGroup(implicit_label, implicit_val).arrange(RIGHT)
        self.place_in_area(implicit_group, "F2", "F5")
        
        self.play(Write(implicit_group))
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # "This circle's equation locks variables together tightly."
        # Stage 4: Draw a circle for the implicit equation and highlight 'x' and 'y' as 'locked' together.
        self.play(self.lecture[3].animate.set_color(YELLOW_IMPLICIT))
        
        # Visualization of the implicit relation
        circle_obj = Circle(radius=1.1, color=YELLOW_IMPLICIT)
        self.place_in_area(circle_obj, "C3", "D4")
        
        # Highlight variables in the equation to emphasize they are "locked"
        highlight_rect_x = SurroundingRectangle(implicit_val[0], color=WHITE, buff=0.1)
        highlight_rect_y = SurroundingRectangle(implicit_val[2], color=WHITE, buff=0.1)
        
        self.play(Create(circle_obj))
        self.play(Create(highlight_rect_x), Create(highlight_rect_y))
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # "How do we find the slope without isolation?"
        # Stage 5: Display the question 'How to find dy/dx?' next to the circle with a pulse effect.
        self.play(self.lecture[4].animate.set_color(YELLOW_IMPLICIT))
        
        question_mobj = Text("How to find dy/dx?", font_size=24, color=WHITE)
        self.place_in_area(question_mobj, "C5", "D6")
        
        self.play(Write(question_mobj))
        # Pulse effect using scale and there_and_back
        self.play(question_mobj.animate.scale(1.2), run_time=0.4, rate_func=there_and_back)
        self.play(question_mobj.animate.scale(1.2), run_time=0.4, rate_func=there_and_back)
        self.wait(2)
