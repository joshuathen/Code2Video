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
        title = "The Feedback Loop: What is Iteration?"
        lines = [
            "Iteration means feeding a function's output back as input.",
            "Imagine a number cycling through the same rule repeatedly.",
            "This feedback loop creates a sequence of evolving values."
        ]
        self.setup_layout(title, lines)
        
        # === Animation for Lecture Line 1 ===
        # Display the word 'Iteration' in #FFFFFF, then replace it with a looping arrow icon.
        self.lecture[0].set_color(WHITE)
        iteration_text = Text("Iteration", color=WHITE)
        # Fix for Issue 23: Move to A3-A4 to avoid overlap with loop_arrow
        self.place_in_area(iteration_text, "A3", "A4", scale_factor=0.8)
        
        self.play(Write(iteration_text))
        self.wait(1)
        
        # Looping arrow icon (Arc with tip)
        loop_arrow = Arc(radius=0.5, start_angle=0, angle=1.5 * PI, color=WHITE).add_tip()
        self.place_in_area(loop_arrow, "B3", "B4", scale_factor=0.8)

        # Transforming instead of simple ReplacementTransform to show evolution
        self.play(ReplacementTransform(iteration_text, loop_arrow))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show a box [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg] labeled 'Rule: $z^2 + c$' in #00FF00, with an input arrow and output arrow looping back.
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color(GREEN)
        
        # Integration of SVG Asset (Issue 19)
        rule_box = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg", color=GREEN)
        
        # Rule label (L022: simple MathTex)
        rule_label_text = Text("Rule: ", font_size=20, color=GREEN)
        rule_label_math = MathTex("z^2 + c", color=GREEN).scale(0.8)
        rule_label = VGroup(rule_label_text, rule_label_math).arrange(RIGHT, buff=0.1)
        
        # Group and place
        rule_group = VGroup(rule_box, rule_label)
        self.place_in_area(rule_group, "C3", "D4", scale_factor=1.2)
        
        # Position label inside or near the box (L003 Area-Positioning)
        # rule_label is already in the group, we'll ensure it's centered relative to the box
        rule_label.move_to(rule_box.get_center())

        # Input arrow
        input_arrow = Arrow(
            start=self.grid["C2"], 
            end=rule_box.get_left(), 
            color=GREEN,
            buff=0.1
        )
        # Output arrow
        output_arrow = Arrow(
            start=rule_box.get_right(), 
            end=self.grid["C5"], 
            color=GREEN,
            buff=0.1
        )
        # Feedback loop arrow (Curved arrow going under)
        feedback_arrow = CurvedArrow(
            start_point=self.grid["C5"],
            end_point=self.grid["C2"],
            angle=-PI,
            color=GREEN
        )

        self.play(
            Create(rule_box),
            Write(rule_label),
            GrowArrow(input_arrow),
            GrowArrow(output_arrow)
        )
        self.play(Create(feedback_arrow))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show a sequence of numbers (e.g., 2, 4, 16...) appearing one by one in #87CEEB.
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color("#87CEEB")
        
        # Create sequence elements
        numbers_list = ["2", "4", "16", "256", "..."]
        num_mobjects = VGroup(*[Text(n, color="#87CEEB") for n in numbers_list]).arrange(RIGHT, buff=0.5)
        # Fix for Issue 24: Adjust area and scale
        self.place_in_area(num_mobjects, "E2", "E5", scale_factor=0.8)
        
        # Animate numbers one by one
        for num in num_mobjects:
            self.play(FadeIn(num, shift=UP * 0.2), run_time=0.4)
            self.wait(0.1)
            
        self.wait(2)
