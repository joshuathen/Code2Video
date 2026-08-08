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

class Section1Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title_text = "The Big Picture: Learning as Fine-Tuning"
        lecture_lines = [
            "Imagine a neural network as a tunable black box.",
            "Learning is adjusting internal knobs to reduce mistakes.",
            "Our goal is making an accurate machine prediction."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_NN = "#ADD8E6"     # Light blue for network/knobs
        COLOR_LOSS = "#FF0000"   # Red for loss/mistakes
        COLOR_SUCCESS = "#00FF00" # Green for correct prediction
        COLOR_FRUIT = "#FFD700"  # Gold for fruit
        
        # Assets
        BOX_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg"
        HAND_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/hand.svg"
        
        # === Animation for Lecture Line 1 ===
        # "Imagine a neural network as a tunable black box."
        self.lecture[0].set_color(COLOR_NN)
        
        # Draw central box [Asset: box.svg]
        nn_box = SVGMobject(BOX_ASSET, color=COLOR_NN)
        self.place_in_area(nn_box, "B2", "E5", scale_factor=2.0)
        
        # Label at A3 (Fixed per Issue 28)
        box_label = Text("Neural Network", font_size=24, color=COLOR_NN)
        self.place_at_grid(box_label, "A3", scale_factor=0.8)
        
        # 5 rotary knobs
        knobs = VGroup()
        knob_positions = ["C2", "C4", "D3", "E2", "E4"]
        for pos in knob_positions:
            knob_base = Circle(radius=0.2, color=COLOR_NN, stroke_width=2)
            knob_indicator = Line(ORIGIN, UP * 0.2, color=COLOR_NN, stroke_width=3)
            knob = VGroup(knob_base, knob_indicator)
            self.place_at_grid(knob, pos, scale_factor=1.0)
            knobs.add(knob)
            
        self.play(
            FadeIn(nn_box),
            Write(box_label),
            LaggedStart(*[Create(k) for k in knobs], lag_ratio=0.1),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Learning is adjusting internal knobs to reduce mistakes."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_LOSS)
        )
        
        # Robotic hand [Asset: hand.svg]
        robotic_hand = SVGMobject(HAND_ASSET, color=WHITE)
        self.place_at_grid(robotic_hand, "D3", scale_factor=0.5)
        robotic_hand.shift(DOWN * 0.5 + RIGHT * 0.2)
        
        # Loss Meter (B6 for label, C6 for frame)
        loss_label = Text("Loss Meter", font_size=20, color=COLOR_LOSS)
        self.place_at_grid(loss_label, "B6", scale_factor=0.8)
        
        meter_frame = Rectangle(height=0.4, width=1.5, color=COLOR_LOSS)
        self.place_at_grid(meter_frame, "C6", scale_factor=1.0)
        
        loss_tracker = ValueTracker(0.8)
        loss_bar = Rectangle(
            height=0.3,
            width=1.4,
            fill_color=COLOR_LOSS,
            fill_opacity=0.8,
            stroke_width=0
        )
        loss_bar.move_to(meter_frame.get_left(), aligned_edge=LEFT).shift(RIGHT * 0.05)
        
        def update_loss_bar(m):
            val = loss_tracker.get_value()
            m.stretch_to_fit_width(max(0.01, val * 1.4), about_edge=LEFT)
            
        loss_bar.add_updater(update_loss_bar)
        
        self.play(
            FadeIn(robotic_hand),
            FadeIn(loss_label),
            Create(meter_frame),
            FadeIn(loss_bar)
        )
        
        # Turning the knob (knobs[2] at D3)
        self.play(
            Rotate(knobs[2][1], angle=-PI/2, about_point=knobs[2][0].get_center()),
            robotic_hand.animate.shift(LEFT * 0.1),
            loss_tracker.animate.set_value(0.4),
            run_time=1.5
        )
        self.play(
            Rotate(knobs[2][1], angle=PI/4, about_point=knobs[2][0].get_center()),
            robotic_hand.animate.shift(RIGHT * 0.2),
            loss_tracker.animate.set_value(0.6),
            run_time=1
        )
        self.play(FadeOut(robotic_hand))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Our goal is making an accurate machine prediction."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_SUCCESS)
        )
        
        # Generic fruit entering the box
        fruit = Circle(radius=0.25, color=COLOR_FRUIT, fill_opacity=0.9)
        fruit_label = Text("?", font_size=20, color=BLACK)
        fruit_group = VGroup(fruit, fruit_label)
        self.place_at_grid(fruit_group, "C1", scale_factor=1.0)
        
        # Prediction at D6 (Fixed per Issue 27)
        prediction_text = Text("Apple", font_size=32, color=COLOR_SUCCESS)
        self.place_at_grid(prediction_text, "D6", scale_factor=1.0)
        
        self.play(FadeIn(fruit_group))
        self.play(
            fruit_group.animate.move_to(nn_box.get_center()),
            run_time=1.5
        )
        
        self.play(
            FadeOut(fruit_group),
            FadeIn(prediction_text),
            loss_tracker.animate.set_value(0.1),
            run_time=1
        )
        
        # Feedback: Success highlight
        self.play(
            prediction_text.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=0.8
        )
        
        self.wait(2)
        
        # Cleanup updaters
        loss_bar.remove_updater(update_loss_bar)
