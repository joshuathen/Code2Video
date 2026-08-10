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

class Section5Scene(TeachingScene):
    def construct(self):
        title_text = "Backpropagation: The Blame Game"
        lecture_lines = [
            "Backpropagation traces the error back to its source.",
            "We calculate which knob contributed most to the mistake.",
            "This \"blame\" is mathematically known as the gradient.",
            "We move backward through the layers using the chain rule.",
            "The robot finds exactly which settings were wrong."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_ERROR = "#FF0000"
        COLOR_KNOB = "#00BFFF"
        COLOR_INPUT = "#32CD32"

        # Elements
        input1 = Circle(radius=0.3, color=COLOR_INPUT, fill_opacity=0.5)
        input2 = Circle(radius=0.3, color=COLOR_INPUT, fill_opacity=0.5)
        label_i1 = Text("Input A", font_size=18)
        label_i2 = Text("Input B", font_size=18)
        
        knob1 = Circle(radius=0.4, color=COLOR_KNOB, fill_opacity=0.3)
        knob2 = Circle(radius=0.4, color=COLOR_KNOB, fill_opacity=0.3)
        label_k1 = Text("Temp", font_size=18)
        label_k2 = Text("Size", font_size=18)
        
        # Dials for knobs
        dial1 = Line(ORIGIN, UP * 0.35, color=WHITE, stroke_width=4)
        dial2 = Line(ORIGIN, UP * 0.35, color=WHITE, stroke_width=4)
        
        output = Circle(radius=0.3, color=WHITE, fill_opacity=0.5)
        label_out = Text("Error", font_size=18)

        # Initial Positioning via grid
        self.place_at_grid(input1, "C1")
        self.place_at_grid(input2, "E1")
        self.place_at_grid(knob1, "C3")
        self.place_at_grid(knob2, "E3")
        self.place_at_grid(output, "D6")
        
        # Relative positioning for labels/dials
        label_i1.next_to(input1, LEFT, buff=0.2)
        label_i2.next_to(input2, LEFT, buff=0.2)
        label_k1.next_to(knob1, UP, buff=0.2)
        label_k2.next_to(knob2, UP, buff=0.2)
        dial1.move_to(knob1.get_center())
        dial2.move_to(knob2.get_center())
        label_out.next_to(output, RIGHT, buff=0.2)

        # Connections
        line1 = Line(input1.get_right(), knob1.get_left(), color=GRAY, stroke_width=2)
        line2 = Line(input2.get_right(), knob2.get_left(), color=GRAY, stroke_width=2)
        line3 = Line(knob1.get_right(), output.get_left(), color=GRAY, stroke_width=2)
        line4 = Line(knob2.get_right(), output.get_left(), color=GRAY, stroke_width=2)

        network_group = VGroup(
            input1, input2, knob1, knob2, output, 
            line1, line2, line3, line4, 
            label_i1, label_i2, label_k1, label_k2, label_out,
            dial1, dial2
        )
        self.add(network_group)

        # === Animation for Lecture Line 1 ===
        # "Backpropagation traces the error back to its source."
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        pulse1 = Dot(output.get_center(), color=COLOR_ERROR, radius=0.15)
        pulse2 = Dot(output.get_center(), color=COLOR_ERROR, radius=0.15)
        
        self.play(FadeIn(pulse1), FadeIn(pulse2))
        self.play(
            pulse1.animate.move_to(knob1.get_center()),
            pulse2.animate.move_to(knob2.get_center()),
            run_time=1, rate_func=linear
        )
        self.play(
            pulse1.animate.move_to(input1.get_center()),
            pulse2.animate.move_to(input2.get_center()),
            run_time=1, rate_func=linear
        )
        self.play(FadeOut(pulse1), FadeOut(pulse2))
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # "We calculate which knob contributed most to the mistake."
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        
        blame1 = Text("80%", font_size=20, color=COLOR_ERROR)
        blame2 = Text("20%", font_size=20, color=COLOR_ERROR)
        self.place_at_grid(blame1, "C4")
        self.place_at_grid(blame2, "E4")
        
        self.play(
            knob1.animate.set_stroke(COLOR_ERROR, width=8),
            knob2.animate.set_stroke(COLOR_ERROR, width=3),
            Write(blame1),
            Write(blame2)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "This \"blame\" is mathematically known as the gradient."
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        
        grad_label = Text("Gradient (∇)", font_size=24, color=COLOR_ERROR).set_weight(BOLD)
        self.place_at_grid(grad_label, "B3")
        
        self.play(FadeIn(grad_label, shift=UP))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "We move backward through the layers using the chain rule."
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(YELLOW))
        
        # Using MathTex for chain rule
        chain_rule = MathTex(r"\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial w}", font_size=24, color=WHITE)
        self.place_at_grid(chain_rule, "A3")
        
        # Red backward arrows
        arr1 = Arrow(output.get_left(), knob1.get_right(), color=COLOR_ERROR, buff=0.1, stroke_width=4)
        arr2 = Arrow(knob1.get_left(), input1.get_right(), color=COLOR_ERROR, buff=0.1, stroke_width=4)
        
        self.play(Write(chain_rule))
        self.play(Create(arr1), Create(arr2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "The robot finds exactly which settings were wrong."
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(YELLOW))
        
        # Drastic adjustment of the 'high blame' knob
        self.play(
            Rotate(dial1, angle=-PI/2, about_point=knob1.get_center()),
            knob1.animate.set_fill(COLOR_ERROR, opacity=0.6),
            blame1.animate.scale(1.2),
            run_time=1.5
        )
        self.wait(2)

        # Final color reset
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
