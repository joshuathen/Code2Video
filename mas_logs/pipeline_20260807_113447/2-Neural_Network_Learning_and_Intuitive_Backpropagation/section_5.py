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

class Section5Scene(TeachingScene):
    def construct(self):
        title = "Intuitive Backpropagation: The Blame Game"
        lines = [
            "Backpropagation sends error signals backward.",
            "It identifies which weights caused the mistake.",
            "The chain rule connects weights to the error.",
            "We blame specific knobs for the wrong guess.",
            "Pulse backward to signal required weight changes."
        ]
        self.setup_layout(title, lines)

        # Assets/Colors
        COLOR_ERROR = "#FF0000"
        COLOR_KNOB = "#ADD8E6"
        COLOR_CHAIN = YELLOW
        KNOB_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/knob.svg"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_ERROR))
        
        error_meter = VGroup(
            RoundedRectangle(height=1, width=0.8, corner_radius=0.1, color=WHITE),
            Text("ERR", font_size=14)
        )
        self.place_in_area(error_meter, "B6", "D6") # Column 6 far right
        self.add(error_meter)

        pulse = Dot(color=COLOR_ERROR, radius=0.1).set_z_index(5)
        
        # Network Nodes
        in_node = Circle(radius=0.25, color=WHITE)
        self.place_at_grid(in_node, "C2")
        h1 = Circle(radius=0.25, color=WHITE)
        self.place_at_grid(h1, "B3")
        h2 = Circle(radius=0.25, color=WHITE)
        self.place_at_grid(h2, "D3")
        out_node = Circle(radius=0.25, color=WHITE)
        self.place_at_grid(out_node, "C4")
        
        nodes = VGroup(in_node, h1, h2, out_node)
        
        # Weights
        w1 = Line(out_node.get_left(), h1.get_right(), color=GRAY)
        w2 = Line(out_node.get_left(), h2.get_right(), color=GRAY)
        w3 = Line(h1.get_left(), in_node.get_right(), color=GRAY)
        w4 = Line(h2.get_left(), in_node.get_right(), color=GRAY)
        
        weights = VGroup(w1, w2, w3, w4)
        self.add(nodes, weights)

        self.play(error_meter[0].animate.set_fill(COLOR_ERROR, opacity=0.5), run_time=0.4)
        self.play(error_meter[0].animate.set_fill(COLOR_ERROR, opacity=0), run_time=0.4)
        
        self.place_at_grid(pulse, "C6")
        self.play(pulse.animate.move_to(out_node.get_center()), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_ERROR)
        )
        
        self.play(
            pulse.animate.move_to(h1.get_center()),
            w1.animate.set_color(COLOR_ERROR),
            run_time=0.6
        )
        self.play(w1.animate.set_color(GRAY), run_time=0.2)
        
        self.play(
            pulse.animate.move_to(h2.get_center()),
            w2.animate.set_color(COLOR_ERROR),
            run_time=0.6
        )
        self.play(w2.animate.set_color(GRAY), run_time=0.2)
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_CHAIN)
        )
        
        chain_label = MathTex(r"\frac{\partial E}{\partial w} = \frac{\partial E}{\partial y} \cdot \frac{\partial y}{\partial w}", font_size=20)
        self.place_in_area(chain_label, 'A3', 'A5', scale_factor=0.8) # Shifted right to avoid C2
        self.play(Write(chain_label))
        
        self.play(
            w3.animate.set_color(COLOR_CHAIN),
            w4.animate.set_color(COLOR_CHAIN),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_KNOB)
        )
        
        knob = SVGMobject(KNOB_ASSET)
        self.place_at_grid(knob, 'E3', scale_factor=0.6)
        
        knob_label = Text("Weight Knob", font_size=16)
        self.place_at_grid(knob_label, 'F3', scale_factor=0.8)
        
        self.play(FadeIn(knob), FadeIn(knob_label))
        self.play(knob.animate.rotate(-PI/2).set_color(COLOR_KNOB), run_time=1.2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(COLOR_KNOB)
        )
        
        self.play(
            pulse.animate.move_to(in_node.get_center()),
            w3.animate.set_color(COLOR_KNOB),
            w4.animate.set_color(COLOR_KNOB),
            run_time=1
        )
        
        self.play(FadeOut(pulse))
        self.wait(2)
