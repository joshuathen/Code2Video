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
        # Setup title and lecture lines
        self.setup_layout("Prerequisite: The MLP Architecture Review", [
            "MLPs consist of two linear layers and an activation.",
            "The first layer W1 projects input to hidden space.",
            "Activation signals if a specific pattern was recognized."
        ])
        
        # Colors based on storyboard requirements
        color_w1 = "#ADD8E6"
        color_act = "#FFFF00"
        color_w2 = "#90EE90"
        color_vector = "#FFFFFF"

        # === Components Preparation ===
        # Input Vector
        input_vec = Vector(RIGHT * 0.7, color=color_vector)
        input_label = Text("Input", font_size=18, color=color_vector)
        
        # Layer W1
        w1_box = Rectangle(height=0.8, width=0.8, color=color_w1, fill_opacity=0.3)
        w1_label = Text("W1", font_size=20, color=color_w1)
        
        # Activation Layer
        act_circle = Circle(radius=0.4, color=color_act, fill_opacity=0.3)
        act_label = Text("ReLU/GELU", font_size=16, color=color_act)
        
        # Layer W2
        w2_box = Rectangle(height=0.8, width=0.8, color=color_w2, fill_opacity=0.3)
        w2_label = Text("W2", font_size=20, color=color_w2)
        
        # Output Vector
        output_vec = Vector(RIGHT * 0.7, color=color_vector)
        output_label = Text("Output", font_size=18, color=color_vector)

        # Positioning using Grid System
        # Issue 34: Fix scale for input_label (0.7) and output_label
        self.place_at_grid(input_vec, "C1")
        self.place_at_grid(input_label, "D1", scale_factor=0.7)
        
        # Issue 33: Fix scale for w1_box (1.2)
        self.place_at_grid(w1_box, "C2", scale_factor=1.2)
        self.place_at_grid(w1_label, "B2")
        
        # Issue 32: Fix scale for act_label (0.5)
        self.place_at_grid(act_circle, "C3")
        self.place_at_grid(act_label, "B3", scale_factor=0.5)
        
        # Scaling W2 for consistency
        self.place_at_grid(w2_box, "C4", scale_factor=1.2)
        self.place_at_grid(w2_label, "B4")
        
        self.place_at_grid(output_vec, "C5")
        self.place_at_grid(output_label, "D5", scale_factor=0.7)

        # Connection Lines - Adjusted buffers for larger boxes
        # Arrow from Vector (tip at 0.35) to W1 (edge at 0.48)
        arrow1 = Line(self.grid["C1"] + RIGHT*0.4, self.grid["C2"] + LEFT*0.5, color=WHITE).add_tip(tip_length=0.1)
        # Arrow from W1 (edge at 0.48) to Act (edge at 0.4)
        arrow2 = Line(self.grid["C2"] + RIGHT*0.5, self.grid["C3"] + LEFT*0.45, color=WHITE).add_tip(tip_length=0.1)
        # Arrow from Act (edge at 0.4) to W2 (edge at 0.48)
        arrow3 = Line(self.grid["C3"] + RIGHT*0.45, self.grid["C4"] + LEFT*0.5, color=WHITE).add_tip(tip_length=0.1)
        # Arrow from W2 (edge at 0.48) to Output Vector
        arrow4 = Line(self.grid["C4"] + RIGHT*0.5, self.grid["C5"] + LEFT*0.4, color=WHITE).add_tip(tip_length=0.1)

        # Pulse Dot for signal flow
        pulse_dot = Dot(color=color_vector, radius=0.08)
        pulse_dot.move_to(self.grid["C1"])

        # === Animation for Lecture Line 1 ===
        # "MLPs consist of two linear layers and an activation."
        # Matching color: Highlight Line 1 in White
        self.lecture[0].set_color(WHITE)
        self.play(
            FadeIn(input_vec), FadeIn(input_label),
            FadeIn(w1_box), FadeIn(w1_label),
            FadeIn(act_circle), FadeIn(act_label),
            FadeIn(w2_box), FadeIn(w2_label),
            FadeIn(output_vec), FadeIn(output_label),
            Create(arrow1), Create(arrow2), Create(arrow3), Create(arrow4),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "The first layer W1 projects input to hidden space."
        # Matching color: Highlight Line 2 in W1 color (#ADD8E6)
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color(color_w1)
        
        self.play(FadeIn(pulse_dot))
        self.play(pulse_dot.animate.move_to(self.grid["C2"]), run_time=1.5)
        self.play(Indicate(w1_box, color=color_w1, scale_factor=1.2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Activation signals if a specific pattern was recognized."
        # Matching color: Highlight Line 3 in Activation color (#FFFF00)
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color(color_act)
        
        # Pulse signal through activation
        self.play(pulse_dot.animate.move_to(self.grid["C3"]), run_time=0.8)
        self.play(
            act_circle.animate.set_fill(color_act, opacity=0.8),
            Flash(act_circle, color=color_act, flash_radius=0.5),
            run_time=0.6
        )
        self.play(act_circle.animate.set_fill(color_act, opacity=0.3))
        
        # Flow through W2 to Output
        self.play(pulse_dot.animate.move_to(self.grid["C4"]), run_time=0.8)
        self.play(Indicate(w2_box, color=color_w2, scale_factor=1.2))
        self.play(pulse_dot.animate.move_to(self.grid["C5"]), run_time=0.8)
        
        self.play(FadeOut(pulse_dot))
        self.wait(3)
