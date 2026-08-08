from manim import *
import numpy as np

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

class Section7Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary: The Weight-Based Encyclopedia", [
            "Models detect keys and retrieve stored values.",
            "Learning is the process of tuning these weights.",
            "Billions of tiny pairs form an internal encyclopedia.",
            "Knowledge is stored directly inside the model's circuitry.",
            "Lex answers correctly because of his internal library."
        ])

        # Colors
        COLOR_FLOW = WHITE
        COLOR_WEIGHTS = YELLOW
        COLOR_POINTS = "#ADD8E6"
        COLOR_CIRCUITS = BLUE_B
        COLOR_LEX = "#00FF00"

        # === Animation for Lecture Line 1 ===
        # Flow diagram: Input -> Key -> Value -> Output.
        # Fixed per Issue 46: input_grp 'B2', w2_grp 'B4', output_grp 'D4'.
        # Also placing w1_grp at 'B3' for continuity.
        
        input_box = Rectangle(width=1.2, height=0.6, color=COLOR_FLOW)
        input_text = Text("Input", font_size=18, color=COLOR_FLOW)
        input_grp = VGroup(input_box, input_text)
        self.place_at_grid(input_grp, "B2", scale_factor=0.6)
        
        w1_box = Rectangle(width=1.2, height=0.6, color=COLOR_FLOW)
        w1_text = MathTex("W_1", font_size=24, color=COLOR_FLOW)
        w1_label = Text("(Key)", font_size=14, color=COLOR_FLOW).next_to(w1_box, DOWN, buff=0.1)
        w1_grp = VGroup(w1_box, w1_text, w1_label)
        self.place_at_grid(w1_grp, "B3", scale_factor=0.6)
        
        w2_box = Rectangle(width=1.2, height=0.6, color=COLOR_FLOW)
        w2_text = MathTex("W_2", font_size=24, color=COLOR_FLOW)
        w2_label = Text("(Value)", font_size=14, color=COLOR_FLOW).next_to(w2_box, DOWN, buff=0.1)
        w2_grp = VGroup(w2_box, w2_text, w2_label)
        self.place_at_grid(w2_grp, "B4", scale_factor=0.6)
        
        output_box = Rectangle(width=1.2, height=0.6, color=COLOR_FLOW)
        output_text = Text("Output", font_size=18, color=COLOR_FLOW)
        output_grp = VGroup(output_box, output_text)
        self.place_at_grid(output_grp, "D4", scale_factor=0.6)

        arrow1 = Arrow(input_grp.get_right(), w1_grp.get_left(), buff=0.1, color=COLOR_FLOW, stroke_width=2)
        arrow2 = Arrow(w1_grp.get_right(), w2_grp.get_left(), buff=0.1, color=COLOR_FLOW, stroke_width=2)
        arrow3 = Arrow(w2_grp.get_bottom(), output_grp.get_top(), buff=0.1, color=COLOR_FLOW, stroke_width=2)

        self.play(self.lecture[0].animate.set_color(COLOR_FLOW))
        self.play(
            Create(input_grp),
            Create(w1_grp),
            Create(w2_grp),
            Create(output_grp),
            GrowArrow(arrow1),
            GrowArrow(arrow2),
            GrowArrow(arrow3),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate W1 and W2 matrices shifting and glowing. Color: #FFFF00.
        self.play(self.lecture[1].animate.set_color(COLOR_WEIGHTS))
        self.play(
            w1_box.animate.set_stroke(COLOR_WEIGHTS, width=6),
            w2_box.animate.set_stroke(COLOR_WEIGHTS, width=6),
            w1_text.animate.set_color(COLOR_WEIGHTS),
            w2_text.animate.set_color(COLOR_WEIGHTS),
            w1_label.animate.set_color(COLOR_WEIGHTS),
            w2_label.animate.set_color(COLOR_WEIGHTS),
        )
        self.play(
            Indicate(w1_grp, color=COLOR_WEIGHTS),
            Indicate(w2_grp, color=COLOR_WEIGHTS),
        )
        self.play(
            w1_box.animate.set_stroke(COLOR_FLOW, width=2),
            w2_box.animate.set_stroke(COLOR_FLOW, width=2),
            w1_text.animate.set_color(COLOR_FLOW),
            w2_text.animate.set_color(COLOR_FLOW),
            w1_label.animate.set_color(COLOR_FLOW),
            w2_label.animate.set_color(COLOR_FLOW),
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show a massive grid of billions of tiny points. Color: #ADD8E6.
        self.play(self.lecture[2].animate.set_color(COLOR_POINTS))
        encyclopedia = VGroup()
        for i in range(5):
            for j in range(6):
                dot = Dot(radius=0.04, color=COLOR_POINTS, fill_opacity=0.8)
                # Spread across grid C2 to E6
                row_char = ["C", "D", "E"]
                col_str = ["2", "3", "4", "5", "6"]
                target_pos = self.grid[f"{row_char[i%3]}{col_str[j%5]}"] + \
                             np.array([np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4), 0])
                dot.move_to(target_pos)
                encyclopedia.add(dot)
        
        self.play(
            FadeOut(input_grp, w1_grp, w2_grp, output_grp, arrow1, arrow2, arrow3),
            FadeIn(encyclopedia, lag_ratio=0.01),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Knowledge is stored directly inside the model's circuitry.
        self.play(self.lecture[3].animate.set_color(WHITE))
        circuits = VGroup()
        for _ in range(25):
            d1 = encyclopedia[np.random.randint(0, len(encyclopedia))]
            d2 = encyclopedia[np.random.randint(0, len(encyclopedia))]
            line = Line(d1.get_center(), d2.get_center(), stroke_width=0.5, color=COLOR_CIRCUITS, stroke_opacity=0.4)
            circuits.add(line)
        
        self.play(Create(circuits), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Lex answers correctly because of his internal library.
        # Fixed per Issue 47: lex 'D3', bubble 'D4'.
        self.play(self.lecture[4].animate.set_color(COLOR_LEX))
        
        # Manual Lex drawing
        lex_head = Square(side_length=0.8, color=COLOR_LEX)
        lex_eye1 = Dot(color=COLOR_LEX).scale(0.6).move_to(lex_head.get_center() + LEFT*0.2 + UP*0.15)
        lex_eye2 = Dot(color=COLOR_LEX).scale(0.6).move_to(lex_head.get_center() + RIGHT*0.2 + UP*0.15)
        lex_mouth = Line(lex_head.get_center() + LEFT*0.2 + DOWN*0.2, lex_head.get_center() + RIGHT*0.2 + DOWN*0.2, color=COLOR_LEX)
        lex_body = Polygon(lex_head.get_bottom(), lex_head.get_bottom() + LEFT*0.4 + DOWN*0.6, lex_head.get_bottom() + RIGHT*0.4 + DOWN*0.6, color=COLOR_LEX)
        lex = VGroup(lex_head, lex_eye1, lex_eye2, lex_mouth, lex_body)
        self.place_at_grid(lex, "D3", scale_factor=0.6) # Scale factor reduced to fit

        bubble = RoundedRectangle(corner_radius=0.1, width=1.4, height=0.6, color=COLOR_LEX)
        self.place_at_grid(bubble, "D4", scale_factor=1.0)
        ans_text = Text("Paris", font_size=20, color=COLOR_LEX)
        ans_text.move_to(bubble.get_center())
        
        self.play(
            FadeOut(encyclopedia, circuits),
            FadeIn(lex),
            run_time=1
        )
        self.play(
            Create(bubble),
            Write(ans_text),
            run_time=1
        )
        self.wait(2)
