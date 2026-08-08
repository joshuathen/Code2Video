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
        # Initializing the layout with the specific title and lecture lines for Section 1
        self.setup_layout(
            "Functions as Space Mappers", 
            [
                "Think of functions as mapping one space to another.",
                "Input points move to new positions on the output.",
                "For f(x) equals 2x, every point moves twice as far."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Reframe the function f(x) not as a static graph, but as a transformation of space.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Drawing two parallel horizontal lines: Input at Row B (y=1.2) and Output at Row E (y=-1.8)
        # Using columns 1 to 6 to span the right-side visual area.
        input_line = Line(start=self.grid["B1"], end=self.grid["B6"], color="#888888")
        output_line = Line(start=self.grid["E1"], end=self.grid["E6"], color="#888888")
        
        input_label = Text("Input", font_size=18, color="#888888")
        output_label = Text("Output", font_size=18, color="#888888")
        
        # Place labels to the right of the lines (Column 6)
        self.place_at_grid(input_label, "B6", scale_factor=0.8).shift(RIGHT * 0.5)
        self.place_at_grid(output_label, "E6", scale_factor=0.8).shift(RIGHT * 0.5)

        self.play(Create(input_line), Create(output_line), FadeIn(input_label), FadeIn(output_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight a point 'x' on the Input line.
        # Per VideoCritic suggestion, we use Column 2 (B2) for x=2 and Column 4 (E4) for f(x)=4.
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        point_x = Dot(point=self.grid["B2"], color="#00FF00")
        label_x = MathTex("x", font_size=24, color="#00FF00")
        # Position label_x above the point (Grid A2) - Resolving Issue #27 and #28
        self.place_at_grid(label_x, "A2", scale_factor=0.8)

        # Draw an initial arrow showing mapping from Input 'x' (B2) to Output 'f(x)' (E4)
        # In f(x)=2x, the distance ratio is 2, mapping Col 2 to Col 4.
        mapping_arrow = Arrow(
            start=self.grid["B2"], 
            end=self.grid["E4"], 
            buff=0.1, 
            color="#00FFFF", 
            stroke_width=3
        )
        point_fx = Dot(point=self.grid["E4"], color="#00FFFF")
        label_fx = MathTex("f(x)", font_size=24, color="#00FFFF")
        # Position label_fx below the point (Grid F4) - Resolving Issue #28
        self.place_at_grid(label_fx, "F4", scale_factor=0.8)

        self.play(FadeIn(point_x), Write(label_x))
        self.play(GrowArrow(mapping_arrow), FadeIn(point_fx), Write(label_fx))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # For f(x)=2x, every point moves twice as far. Show multiple arrows.
        self.play(self.lecture[2].animate.set_color("#00FFFF"))

        # Define points mapping x to 2x using distance from an origin at imaginary Col 0.
        # B1 (dist 1) -> E2 (dist 2)
        # B2 (dist 2) -> E4 (dist 4)
        # B3 (dist 3) -> E6 (dist 6)
        points_input = [self.grid["B1"], self.grid["B2"], self.grid["B3"]]
        points_output = [self.grid["E2"], self.grid["E4"], self.grid["E6"]]
        
        arrows = VGroup(*[
            Arrow(start=pi, end=po, buff=0.1, color="#00FFFF", stroke_width=2) 
            for pi, po in zip(points_input, points_output)
        ])
        
        # Remove the previous single arrow/dot to avoid duplication when showing the group
        self.remove(mapping_arrow, point_fx)
        
        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.3),
            run_time=2
        )
        self.wait(2)
