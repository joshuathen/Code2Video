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

class Section2Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        lecture_lines = [
            "Inputs enter the network and travel from left to right.",
            "Each input is multiplied by a weight.",
            "Weighted values are summed at each neuron.",
            "An activation function acts as a gatekeeper for signals.",
            "The final output is the network's current prediction."
        ]
        self.setup_layout("The Forward Pass: Making a Guess", lecture_lines)
        
        # Colors from storyboard and prompt requirements
        input_color = "#ADD8E6"
        weight_color = "#FF69B4"
        gatekeeper_color = "#FF4500"
        output_color = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Inputs enter the network and travel from left to right.
        self.lecture[0].set_color(input_color)
        
        input_1 = Text("350", font_size=36, color=input_color)
        input_2 = Text("10", font_size=36, color=input_color)
        self.place_at_grid(input_1, "B1")
        self.place_at_grid(input_2, "E1")
        
        label_1 = Text("Temp (°F)", font_size=18, color=input_color)
        label_2 = Text("Time (min)", font_size=18, color=input_color)
        label_1.next_to(input_1, UP, buff=0.1)
        label_2.next_to(input_2, UP, buff=0.1)

        self.play(FadeIn(input_1, label_1), FadeIn(input_2, label_2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Each input is multiplied by a weight.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(weight_color)
        
        gatekeeper = Circle(radius=0.5, color=gatekeeper_color, fill_opacity=0.3)
        # Fix Issue 38: Use place_in_area for gatekeeper
        self.place_in_area(gatekeeper, 'B4', 'D4', scale_factor=0.8)
        
        gatekeeper_label = Text("Gatekeeper", font_size=20, color=gatekeeper_color)
        gatekeeper_label.next_to(gatekeeper, UP, buff=0.2)
        
        line_1 = Line(input_1.get_right(), gatekeeper.get_left(), color=weight_color)
        line_2 = Line(input_2.get_right(), gatekeeper.get_left(), color=weight_color)
        
        weight_label_1 = Text("x Weight A", font_size=20, color=weight_color)
        weight_label_2 = Text("x Weight B", font_size=20, color=weight_color)
        
        # Fix Issue 39: Use place_in_area for multi-word weight labels
        self.place_in_area(weight_label_1, 'B2', 'B3', scale_factor=0.7)
        self.place_in_area(weight_label_2, 'E2', 'E3', scale_factor=0.7)

        self.play(Create(line_1), Create(line_2))
        self.play(FadeIn(weight_label_1), FadeIn(weight_label_2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Weighted values are summed at each neuron.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(gatekeeper_color)
        
        dot_1 = Dot(input_1.get_center(), color=WHITE)
        dot_2 = Dot(input_2.get_center(), color=WHITE)
        
        self.play(
            dot_1.animate.move_to(gatekeeper.get_center()),
            dot_2.animate.move_to(gatekeeper.get_center()),
            run_time=1.5
        )
        self.remove(dot_1, dot_2)
        
        sum_label = Text("+", font_size=40, color=WHITE)
        sum_label.move_to(gatekeeper.get_center())
        self.play(FadeIn(sum_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # An activation function acts as a gatekeeper for signals.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(gatekeeper_color)
        
        self.remove(sum_label)
        self.play(FadeIn(gatekeeper), FadeIn(gatekeeper_label))
        # Pulse animation
        self.play(gatekeeper.animate.scale(1.2).set_color(WHITE), run_time=0.3)
        self.play(gatekeeper.animate.scale(1/1.2).set_color(gatekeeper_color), run_time=0.3)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The final output is the network's current prediction.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(output_color)
        
        # Integrate Asset: cookie.svg (Issue 19)
        cookie_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cookie.svg"
        prediction_icon = SVGMobject(cookie_path)
        prediction_text = Text("Burnt Cookie", font_size=24, color=output_color)
        prediction_group = VGroup(prediction_icon, prediction_text).arrange(DOWN, buff=0.2)
        
        # Fix Issue 37: Use place_in_area for prediction group
        self.place_in_area(prediction_group, 'C5', 'C6', scale_factor=0.8)
        
        arrow_out = Arrow(gatekeeper.get_right(), prediction_group.get_left(), color=WHITE)
        
        self.play(Create(arrow_out))
        self.play(FadeIn(prediction_group))
        self.wait(2)
        
        self.lecture[4].set_color(WHITE)
        self.wait(2)
