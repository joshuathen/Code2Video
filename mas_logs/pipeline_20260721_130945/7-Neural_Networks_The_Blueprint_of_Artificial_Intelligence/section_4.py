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

class Section4Scene(TeachingScene):
    def construct(self):
        title_text = "Layered Architecture: The Assembly Line"
        lecture_lines = [
            "Neurons organize into input, hidden, and output layers.",
            "Data flows forward, becoming more abstract at each stage.",
            "Raw pixels transform into complex patterns like digit shapes."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        color_input = BLUE_C
        color_hidden = GREEN_C
        color_output = RED_C
        color_line = GRAY_E
        color_pulse = YELLOW_A

        # Assets
        asset_seven = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/seven.svg"
        asset_pixel = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/pixel.svg"

        # === Animation for Lecture Line 1 ===
        # Neurons organize into input, hidden, and output layers.
        self.lecture[0].set_color(color_input)
        
        # 1. Digit 7 and Pixel Grid (Visual context for the architecture)
        # Using [Asset: .../pixel.svg] and [Asset: .../seven.svg]
        pixel_grid = VGroup(*[SVGMobject(asset_pixel).scale(0.12) for _ in range(16)]).arrange_in_grid(4, 4, buff=0.05)
        digit_7_input = SVGMobject(asset_seven).scale(0.5).set_color(WHITE)
        pixel_digit = VGroup(pixel_grid, digit_7_input)
        # Fix Issue 30: Move pixel_digit to C1-D1, scale 0.7
        self.place_in_area(pixel_digit, "C1", "D1", scale_factor=0.7)

        # 2. Network Layers
        input_layer = VGroup(*[Circle(radius=0.12, color=color_input, fill_opacity=0.7) for _ in range(3)]).arrange(DOWN, buff=0.4)
        hidden_layer = VGroup(*[Circle(radius=0.12, color=color_hidden, fill_opacity=0.7) for _ in range(4)]).arrange(DOWN, buff=0.3)
        output_layer = VGroup(*[Circle(radius=0.12, color=color_output, fill_opacity=0.7) for _ in range(2)]).arrange(DOWN, buff=0.6)
        
        # Fix Issue 31: Relocate layers and labels to improve spacing
        self.place_at_grid(input_layer, "C2")
        self.place_at_grid(hidden_layer, "C4")
        self.place_at_grid(output_layer, "C6")
        
        # Labels
        lbl_in = Text("Input", font_size=18, color=color_input)
        lbl_hid = Text("Hidden", font_size=18, color=color_hidden)
        lbl_out = Text("Output", font_size=18, color=color_output)
        
        self.place_at_grid(lbl_in, "B2", scale_factor=0.7)
        self.place_at_grid(lbl_hid, "B4", scale_factor=0.7)
        self.place_at_grid(lbl_out, "B6", scale_factor=0.7)
        
        self.play(
            FadeIn(pixel_digit),
            FadeIn(input_layer), FadeIn(lbl_in),
            FadeIn(hidden_layer), FadeIn(lbl_hid),
            FadeIn(output_layer), FadeIn(lbl_out),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Data flows forward, becoming more abstract at each stage.
        self.lecture[1].set_color(color_hidden)
        
        # Connections
        conn_in_hid = VGroup()
        for i_n in input_layer:
            for h_n in hidden_layer:
                conn_in_hid.add(Line(i_n.get_center(), h_n.get_center(), stroke_width=1, color=color_line))
                
        conn_hid_out = VGroup()
        for h_n in hidden_layer:
            for o_n in output_layer:
                conn_hid_out.add(Line(h_n.get_center(), o_n.get_center(), stroke_width=1, color=color_line))
        
        self.play(Create(conn_in_hid), Create(conn_hid_out), run_time=1.5)
        
        # Pulse Flow - visualizing abstract transformation
        # Select representative paths for the pulse
        pulse_paths = [
            [input_layer[0], hidden_layer[1], output_layer[0]],
            [input_layer[1], hidden_layer[2], output_layer[1]],
            [input_layer[2], hidden_layer[3], output_layer[0]]
        ]
        
        pulse_animations = []
        for path in pulse_paths:
            p_dot = Dot(radius=0.05, color=color_pulse)
            p_dot.move_to(path[0].get_center())
            pulse_animations.append(
                Succession(
                    FadeIn(p_dot, scale=0.5),
                    p_dot.animate.move_to(path[1].get_center()),
                    p_dot.animate.move_to(path[2].get_center()),
                    FadeOut(p_dot),
                    run_time=2
                )
            )

        self.play(AnimationGroup(*pulse_animations, lag_ratio=0.3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Raw pixels transform into complex patterns like digit shapes.
        self.lecture[2].set_color(color_output)
        
        # Connection from pixel grid to input to signify data entry
        pixel_conns = VGroup()
        for i_n in input_layer:
            pixel_conns.add(Line(pixel_digit.get_right(), i_n.get_left(), stroke_width=1, color=color_line, stroke_opacity=0.5))
            
        self.play(Create(pixel_conns))
        
        # Highlight transformation
        self.play(
            Indicate(pixel_digit),
            Indicate(hidden_layer),
            Indicate(output_layer),
            run_time=2
        )
        
        # Final classification result using [Asset: .../seven.svg]
        # Fix Issue 32: Place result at E6 to align with output layer shift
        res_text = Text("Result:", font_size=20, color=color_output)
        digit_7_output = SVGMobject(asset_seven).scale(0.4).set_color(color_output)
        res_group = VGroup(res_text, digit_7_output).arrange(RIGHT, buff=0.2)
        
        self.place_at_grid(res_group, "E6", scale_factor=0.8)
        
        self.play(Write(res_text), FadeIn(digit_7_output, shift=UP*0.2))
        
        self.wait(2)
