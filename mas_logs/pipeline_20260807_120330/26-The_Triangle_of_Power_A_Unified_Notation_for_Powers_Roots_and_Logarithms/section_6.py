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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup the layout with section title and lecture lines
        # Storyboard Lines:
        # 1. "To find a logarithm, look at the top vertex."
        # 2. "How high must the base raise to reach result?"
        # 3. "Log base ten of one thousand equals three."
        self.setup_layout("Operation 3: Solving the Logarithm", [
            "To find a logarithm, look at the top vertex.",
            "How high must the base raise to reach result?",
            "Log base ten of one thousand equals three."
        ])

        # Colors for the Triangle of Power elements
        BASE_COLOR = "#87CEEB"    # Sky Blue
        RESULT_COLOR = "#98FB98"  # Pale Green
        EXPONENT_COLOR = "#E74C3C" # Red/Orange for focus

        # Grid positions for the triangle vertices (Updated per VideoCritic issue #41)
        # We'll use C5 (Top), E4 (Bottom-Left), E6 (Bottom-Right)
        top_pos = self.grid["C5"]
        bl_pos = self.grid["E4"]
        br_pos = self.grid["E6"]

        # Triangle lines with buffs to avoid overlapping the vertex labels
        l1 = Line(bl_pos, top_pos, buff=0.4, color=WHITE)
        l2 = Line(br_pos, top_pos, buff=0.4, color=WHITE)
        l3 = Line(bl_pos, br_pos, buff=0.5, color=WHITE)
        triangle_group = VGroup(l1, l2, l3)

        # Mobjects for the values
        base_val = MathTex("10", color=BASE_COLOR)
        result_val = MathTex("1000", color=RESULT_COLOR)
        question_mark = MathTex("?", color=EXPONENT_COLOR)
        exponent_val = MathTex("3", color=EXPONENT_COLOR)

        # Position mobjects at grid points (Updated per VideoCritic issue #41)
        self.place_at_grid(base_val, "E4", scale_factor=1.1)
        self.place_at_grid(result_val, "E6", scale_factor=1.0)
        self.place_at_grid(question_mark, "C5", scale_factor=1.1)
        self.place_at_grid(exponent_val, "C5", scale_factor=1.1)

        # === Animation for Lecture Line 1 ===
        # "To find a logarithm, look at the top vertex."
        # Storyboard: "In the Triangle, bold the Base and Result values."
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(
            Create(triangle_group),
            Write(base_val),
            Write(result_val),
            Write(question_mark)
        )
        # Bold/Highlight Base and Result as per storyboard
        self.play(
            base_val.animate.set_stroke(width=2).scale(1.1),
            result_val.animate.set_stroke(width=2).scale(1.1),
            Indicate(question_mark, color=WHITE) # Drawing attention to the top vertex
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "How high must the base raise to reach result?"
        # Storyboard: "Show a flashing '?' at the Exponent vertex."
        self.play(self.lecture[1].animate.set_color(WHITE))
        # Flash the question mark using a loop or succession
        self.play(
            Succession(
                question_mark.animate.set_opacity(0.3),
                question_mark.animate.set_opacity(1.0),
                question_mark.animate.set_opacity(0.3),
                question_mark.animate.set_opacity(1.0),
                question_mark.animate.set_opacity(0.3),
                question_mark.animate.set_opacity(1.0),
            ),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Log base ten of one thousand equals three."
        # Storyboard: "Transform the '?' into the number '3' in #E74C3C."
        self.play(self.lecture[2].animate.set_color(EXPONENT_COLOR))
        
        # Transform "?" to "3"
        self.play(Transform(question_mark, exponent_val))
        self.play(Indicate(question_mark, color=EXPONENT_COLOR))
        
        self.wait(3)
