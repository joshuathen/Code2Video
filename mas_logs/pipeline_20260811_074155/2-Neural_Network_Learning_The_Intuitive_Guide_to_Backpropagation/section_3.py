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
        self.setup_layout("The Forward Pass: Making a Guess", [
            "The forward pass is the network's first guess.",
            "It takes cake temperature and size as inputs.",
            "Weights multiply these inputs to calculate their importance.",
            "Adding the bias produces a final time prediction.",
            "Robo-Chef now has a guess for the baking time."
        ])
        
        # === Animation for Lecture Line 1 ===
        # "The forward pass is the network's first guess."
        self.lecture[0].set_color(WHITE)
        
        # Neuron
        neuron = Circle(radius=0.4, color=WHITE, stroke_width=4)
        self.place_at_grid(neuron, 'C3')
        neuron_label = Text("Neuron", font_size=18, color=WHITE).next_to(neuron, DOWN, buff=0.1)
        
        # Input locations
        input1_point = self.grid['B1']
        input2_point = self.grid['D1']
        
        # Connections
        arrow1 = Arrow(start=input1_point, end=neuron.get_left(), buff=0.1, color=WHITE)
        arrow2 = Arrow(start=input2_point, end=neuron.get_left(), buff=0.1, color=WHITE)
        
        self.play(Create(neuron), Write(neuron_label))
        self.play(GrowArrow(arrow1), GrowArrow(arrow2))
        
        # Pulse arrows
        self.play(
            arrow1.animate.set_stroke(width=10),
            arrow2.animate.set_stroke(width=10),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "It takes cake temperature and size as inputs."
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color(BLUE_C)
        
        temp_input = VGroup(
            RoundedRectangle(corner_radius=0.1, height=0.6, width=1.4, color=BLUE_C),
            Text("Temp", font_size=16, color=BLUE_C)
        )
        self.place_at_grid(temp_input, 'B1')
        
        size_input = VGroup(
            RoundedRectangle(corner_radius=0.1, height=0.6, width=1.4, color=BLUE_C),
            Text("Size", font_size=16, color=BLUE_C)
        )
        self.place_at_grid(size_input, 'D1')
        
        self.play(FadeIn(temp_input), FadeIn(size_input))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Weights multiply these inputs to calculate their importance."
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color(ORANGE)
        
        # Dials for Weights
        weight1_dial = VGroup(
            Circle(radius=0.25, color=ORANGE, stroke_width=3),
            Line(ORIGIN, UP * 0.25, color=ORANGE, stroke_width=4)
        )
        self.place_at_grid(weight1_dial, 'B2')
        w1_label = Text("Weight 1", font_size=14, color=ORANGE).next_to(weight1_dial, UP, buff=0.1)
        
        weight2_dial = VGroup(
            Circle(radius=0.25, color=ORANGE, stroke_width=3),
            Line(ORIGIN, UP * 0.25, color=ORANGE, stroke_width=4)
        )
        self.place_at_grid(weight2_dial, 'D2')
        w2_label = Text("Weight 2", font_size=14, color=ORANGE).next_to(weight2_dial, UP, buff=0.1)
        
        # Data dots moving through weights
        dot1 = Dot(point=input1_point, color=YELLOW, radius=0.08)
        dot2 = Dot(point=input2_point, color=YELLOW, radius=0.08)
        
        self.play(
            FadeIn(weight1_dial), FadeIn(weight2_dial),
            Write(w1_label), Write(w2_label)
        )
        
        # Animate dials rotating while dots pass
        self.play(
            Rotate(weight1_dial[1], angle=TAU*2, about_point=weight1_dial[0].get_center()),
            Rotate(weight2_dial[1], angle=-TAU*2, about_point=weight2_dial[0].get_center()),
            dot1.animate.move_to(neuron.get_center()),
            dot2.animate.move_to(neuron.get_center()),
            run_time=3,
            rate_func=linear
        )
        self.remove(dot1, dot2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Adding the bias produces a final time prediction."
        self.lecture[2].set_color(GRAY)
        self.lecture[3].set_color(YELLOW)
        
        bias_label = Text("+ Bias", font_size=18, color=PINK)
        self.place_at_grid(bias_label, 'C2')
        
        # Prediction Box
        prediction_box = VGroup(
            Rectangle(height=0.8, width=2.4, color=YELLOW, stroke_width=4),
            Text("Prediction: 20 min", font_size=18, color=YELLOW)
        )
        self.place_at_grid(prediction_box, 'C5')
        
        # Arrow to prediction
        arrow_to_pred = Arrow(start=neuron.get_right(), end=prediction_box.get_left(), color=WHITE, buff=0.1)
        
        self.play(Write(bias_label))
        self.play(GrowArrow(arrow_to_pred))
        self.play(FadeIn(prediction_box, shift=RIGHT))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Robo-Chef now has a guess for the baking time."
        self.lecture[3].set_color(GRAY)
        self.lecture[4].set_color(YELLOW)
        
        # Highlight prediction box
        self.play(
            prediction_box.animate.scale(1.1).set_color(WHITE),
            rate_func=there_and_back,
            run_time=1
        )
        self.wait(2)
