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
        # Data
        title_text = "Visualizing Non-Linear Transformation (f(x) = x²)"
        lecture_lines = [
            "Observe the transformation for the function x squared.",
            "At x equals one, the scaling factor is two.",
            "At x equals three, the scaling increases to six.",
            "The stretching power changes across the input line.",
            "Larger inputs result in much more intense stretching."
        ]
        
        # Setup
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_TEXT = "#FFFFFF"
        COLOR_STRETCH = "#FF4500"
        COLOR_IN = "#00FF00"
        COLOR_OUT = "#00BFFF"
        
        # === Animation for Lecture Line 1 ===
        # Observe the transformation for the function x squared.
        self.lecture[0].set_color(YELLOW)
        
        equation = MathTex("f(x) = x^2", color=COLOR_TEXT)
        self.place_in_area(equation, "A3", "A4", scale_factor=0.8)
        
        # Input Line
        input_line = NumberLine(x_range=[0, 4, 1], length=5, include_numbers=True, font_size=16)
        self.place_in_area(input_line, "C1", "C6")
        input_label = Text("Input Line (x)", font_size=16, color=WHITE)
        self.place_in_area(input_label, "B1", "B3", scale_factor=0.8)
        
        # Output Line
        output_line = NumberLine(x_range=[0, 16, 4], length=5, include_numbers=True, font_size=16)
        self.place_in_area(output_line, "E1", "E6")
        output_label = Text("Output Line (f(x))", font_size=16, color=WHITE)
        self.place_in_area(output_label, "F1", "F3", scale_factor=0.8)

        self.play(Write(equation))
        self.play(Create(input_line), FadeIn(input_label))
        self.play(Create(output_line), FadeIn(output_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # At x equals one, the scaling factor is two.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        dx = 0.2
        x1 = 1.0
        
        # Interval at x=1
        rect_in1 = Line(
            input_line.n2p(x1 - dx/2), 
            input_line.n2p(x1 + dx/2), 
            color=COLOR_IN, stroke_width=10
        )
        label_in1 = MathTex("dx=0.2", font_size=18, color=COLOR_IN)
        label_in1.next_to(rect_in1, UP, buff=0.1)
        
        # Mapped interval
        y1_start, y1_end = (x1 - dx/2)**2, (x1 + dx/2)**2
        rect_out1 = Line(
            output_line.n2p(y1_start),
            output_line.n2p(y1_end),
            color=COLOR_OUT, stroke_width=10
        )
        df1 = y1_end - y1_start
        label_out1 = MathTex(f"df={df1:.1f}", font_size=18, color=COLOR_OUT)
        label_out1.next_to(rect_out1, DOWN, buff=0.1)
        
        # Connection
        arrow1 = Arrow(rect_in1.get_bottom(), rect_out1.get_top(), buff=0.1, color=WHITE, stroke_width=2)
        scale_txt1 = Text("Scale: 2x", font_size=18, color=COLOR_STRETCH)
        scale_txt1.next_to(arrow1, RIGHT, buff=0.1)

        self.play(Create(rect_in1), FadeIn(label_in1))
        self.play(GrowArrow(arrow1))
        self.play(Create(rect_out1), FadeIn(label_out1))
        self.play(Write(scale_txt1))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # At x equals three, the scaling increases to six.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        x3 = 3.0
        rect_in3 = Line(
            input_line.n2p(x3 - dx/2), 
            input_line.n2p(x3 + dx/2), 
            color=COLOR_IN, stroke_width=10
        )
        label_in3 = MathTex("dx=0.2", font_size=18, color=COLOR_IN)
        label_in3.next_to(rect_in3, UP, buff=0.1)
        
        y3_start, y3_end = (x3 - dx/2)**2, (x3 + dx/2)**2
        rect_out3 = Line(
            output_line.n2p(y3_start),
            output_line.n2p(y3_end),
            color=COLOR_OUT, stroke_width=10
        )
        df3 = y3_end - y3_start
        label_out3 = MathTex(f"df={df3:.1f}", font_size=18, color=COLOR_OUT)
        label_out3.next_to(rect_out3, DOWN, buff=0.1)
        
        arrow3 = Arrow(rect_in3.get_bottom(), rect_out3.get_top(), buff=0.1, color=WHITE, stroke_width=2)
        scale_txt3 = Text("Scale: 6x", font_size=18, color=COLOR_STRETCH)
        scale_txt3.next_to(arrow3, RIGHT, buff=0.1)

        self.play(
            FadeOut(rect_in1, label_in1, arrow1, rect_out1, label_out1, scale_txt1),
            Create(rect_in3), FadeIn(label_in3)
        )
        self.play(GrowArrow(arrow3))
        self.play(Create(rect_out3), FadeIn(label_out3))
        self.play(Write(scale_txt3))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # The stretching power changes across the input line.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        input_line_gradient = input_line.copy()
        input_line_gradient.set_color_by_gradient(WHITE, COLOR_STRETCH)
        
        self.play(Transform(input_line, input_line_gradient))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # Larger inputs result in much more intense stretching.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Zoom on output line around the region of interest
        output_zoom_group = VGroup(output_line, rect_out3, label_out3, output_label)
        
        # Calculate center of visualization area to move the zoomed group into
        tl_pos = self.grid["B1"]
        br_pos = self.grid["F6"]
        center_viz = np.array([(tl_pos[0] + br_pos[0]) / 2, (tl_pos[1] + br_pos[1]) / 2, 0])

        self.play(
            output_zoom_group.animate.scale(2.5).move_to(center_viz),
            FadeOut(arrow3, rect_in3, label_in3, input_line, input_label, equation, scale_txt3),
            run_time=2
        )
        
        self.play(Indicate(rect_out3, color=COLOR_OUT))
        self.wait(3)
        
        # Final cleanup
        self.lecture[4].set_color(WHITE)
        self.wait(1)
