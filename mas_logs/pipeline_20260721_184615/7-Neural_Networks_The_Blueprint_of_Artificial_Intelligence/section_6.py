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
        title = "The Feedback Loop: How Networks Learn"
        lines = [
            "- Loss measures the difference between prediction and reality.",
            "- We use this error to refine the network's performance.",
            "- The algorithm slightly adjusts weights to reduce future mistakes.",
            "- This iterative process is how a machine actually learns.",
            "- With enough practice, the model achieves high accuracy."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Line 1 color change: Red for Loss
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        
        # Prediction node - Moved to C5 per Issue 38
        pred_node = Circle(radius=0.4, color="#FFFFFF", stroke_width=2)
        self.place_at_grid(pred_node, 'C5')
        
        node_label = Text("Prediction", font_size=16, color="#FFFFFF")
        node_label.next_to(pred_node, UP, buff=0.2)
        
        # Red X mark using Cross mobject
        x_mark = Cross(pred_node, stroke_color="#FF0000", stroke_width=8)
        
        self.play(Create(pred_node), Write(node_label))
        self.play(Create(x_mark))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Line 2 color change: Red for Error
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        
        # Error label - Moved to area D5-E6 per Issue 37/44
        error_label = Text("Error (Loss)", font_size=20, color="#FF0000")
        self.place_in_area(error_label, 'D5', 'E6', scale_factor=0.7)
        
        self.play(Write(error_label))
        self.play(Indicate(x_mark, color="#FF0000"))
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # Line 3 color change: Cyan for adjustment
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        
        # Input nodes
        in1 = Circle(radius=0.2, color="#FFFFFF", stroke_width=2)
        self.place_at_grid(in1, 'B2')
        in2 = Circle(radius=0.2, color="#FFFFFF", stroke_width=2)
        self.place_at_grid(in2, 'D2')
        
        # Weights (lines) - Adjusted to C5
        w1_line = Line(self.grid['B2'], self.grid['C5'], color="#FFFFFF", stroke_width=2, buff=0.2)
        w2_line = Line(self.grid['D2'], self.grid['C5'], color="#FFFFFF", stroke_width=2, buff=0.2)
        
        # Labels for weights - using simple MathTex for subscripts
        w1_val = DecimalNumber(0.80, num_decimal_places=2, font_size=22, color="#FFFFFF")
        w1_tex = Text("w1 =", font_size=18, color="#FFFFFF") # Using Text instead of MathTex to be safe (L022)
        w1_label = VGroup(w1_tex, w1_val).arrange(RIGHT, buff=0.1)
        w1_label.next_to(w1_line, UP, buff=0.1)
        
        w2_val = DecimalNumber(0.30, num_decimal_places=2, font_size=22, color="#FFFFFF")
        w2_tex = Text("w2 =", font_size=18, color="#FFFFFF")
        w2_label = VGroup(w2_tex, w2_val).arrange(RIGHT, buff=0.1)
        w2_label.next_to(w2_line, DOWN, buff=0.1)
        
        # Adjustment arrows
        arrow1 = Arrow(start=UP, end=DOWN, color="#00FFFF", buff=0, stroke_width=4, max_tip_length_to_length_ratio=0.3).scale(0.3)
        arrow1.next_to(w1_label, RIGHT, buff=0.1)
        arrow2 = Arrow(start=DOWN, end=UP, color="#00FFFF", buff=0, stroke_width=4, max_tip_length_to_length_ratio=0.3).scale(0.3)
        arrow2.next_to(w2_label, RIGHT, buff=0.1)

        self.play(
            Create(in1), Create(in2), 
            Create(w1_line), Create(w2_line),
            Write(w1_label), Write(w2_label)
        )
        self.play(GrowArrow(arrow1), GrowArrow(arrow2))
        
        # Weight adjustment animation
        self.play(
            w1_val.animate.set_value(0.75).set_color("#00FFFF"),
            w2_val.animate.set_value(0.35).set_color("#00FFFF"),
            run_time=1.5
        )
        self.wait(2.0)

        # === Animation for Lecture Line 4 ===
        # Line 4 color change: Cyan for iterative process
        self.play(self.lecture[3].animate.set_color("#00FFFF"))
        
        # Group network for indication
        network_elements = VGroup(in1, in2, w1_line, w2_line, w1_label, w2_label, pred_node)
        self.play(Indicate(network_elements, color="#00FFFF", scale_factor=1.05))
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        # Line 5 color change: Green for High Accuracy
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        
        # Check mark construction
        check_part1 = Line(start=LEFT * 0.2 + DOWN * 0.1, end=DOWN * 0.3, color="#00FF00", stroke_width=8)
        check_part2 = Line(start=DOWN * 0.3, end=RIGHT * 0.4 + UP * 0.3, color="#00FF00", stroke_width=8)
        check_mark = VGroup(check_part1, check_part2).move_to(x_mark.get_center())
        
        # Transform red X into green check
        self.play(FadeOut(error_label), FadeOut(arrow1), FadeOut(arrow2))
        self.play(ReplacementTransform(x_mark, check_mark))
        
        # Indicate success on prediction node
        self.play(
            node_label.animate.set_color("#00FF00"),
            pred_node.animate.set_color("#00FF00")
        )
        self.wait(2.0)
