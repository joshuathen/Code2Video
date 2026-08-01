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
        # Initial Setup with provided title and lecture lines
        title_text = "The Strategy: The Four-Step Recipe"
        lecture_lines = [
            "First, differentiate both sides of the equation normally.",
            "Attach a dy/dx whenever you differentiate a y term.",
            "Next, group all terms containing dy/dx on one side.",
            "Factor out dy/dx to isolate it from the rest.",
            "Finally, solve for dy/dx to find the derivative."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Step: Differentiate both sides normally.
        # Action: Display d/dx(x^2) + d/dx(y^2) = d/dx(25) in white (#FFFFFF).
        self.play(self.lecture[0].animate.set_color(YELLOW))
        step1_eq = Text("d/dx(x^2) + d/dx(y^2) = d/dx(25)", color="#FFFFFF", font_size=24)
        self.place_in_area(step1_eq, "C1", "D6", scale_factor=0.8)
        self.play(Write(step1_eq))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Step: Attach a dy/dx whenever you differentiate a y term.
        # Action: Show 2x + 2y(dy/dx) = 0, with dy/dx highlighted in orange (#FFA500).
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFA500")
        )
        step2_eq = Text("2x + 2y(dy/dx) = 0", color="#FFFFFF", font_size=24, t2c={"(dy/dx)": "#FFA500"})
        self.place_in_area(step2_eq, "C1", "D6", scale_factor=0.8)
        self.play(ReplacementTransform(step1_eq, step2_eq))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Step: Group all terms containing dy/dx on one side.
        # Action: Move the 2x term to the right side to get 2y(dy/dx) = -2x.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        step3_eq = Text("2y(dy/dx) = -2x", color="#FFFFFF", font_size=24, t2c={"(dy/dx)": "#FFA500"})
        self.place_in_area(step3_eq, "C1", "D6", scale_factor=0.8)
        self.play(ReplacementTransform(step2_eq, step3_eq))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # Step: Factor out dy/dx to isolate it.
        # Action: Divide both sides by 2y to isolate the dy/dx term.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        step4_eq = Text("dy/dx = -2x / 2y", color="#FFFFFF", font_size=24, t2c={"dy/dx": "#FFA500"})
        self.place_in_area(step4_eq, "C1", "D6", scale_factor=0.8)
        self.play(ReplacementTransform(step3_eq, step4_eq))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # Step: Solve for dy/dx to find the derivative.
        # Action: Highlight the final solution dy/dx = -x/y in yellow (#FFFF00).
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color("#FFFF00")
        )
        final_eq = Text("dy/dx = -x/y", color="#FFFF00", font_size=28)
        self.place_in_area(final_eq, "C1", "D6", scale_factor=1.2)
        self.play(ReplacementTransform(step4_eq, final_eq))
        self.wait(3)
