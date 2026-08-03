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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Loss Function: Measuring the Mistake", [
            "To improve, we must measure how wrong we are.",
            "The loss function calculates the distance from the truth.",
            "High loss means a large error in the prediction."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display 'Prediction' (#FF0000) and 'Target' (#00FF00) as two distinct points on a vertical scale.
        self.lecture[0].set_color("#FF0000")
        
        # Vertical Scale (Line from A3 to F3)
        scale_line = Line(self.grid["A3"], self.grid["F3"], color=GRAY_B)
        
        prediction_dot = Dot(color="#FF0000").scale(1.5)
        self.place_at_grid(prediction_dot, "D3")
        prediction_label = Text("Prediction", font_size=20, color="#FF0000")
        # Issue 27 fix: Adjusted scale_factor to 0.8 to prevent vertical crowding.
        self.place_at_grid(prediction_label, "D4", scale_factor=0.8)
        
        target_dot = Dot(color="#00FF00").scale(1.5)
        self.place_at_grid(target_dot, "B3")
        target_label = Text("Target", font_size=20, color="#00FF00")
        # Issue 27 fix: Adjusted scale_factor to 0.8 to prevent vertical crowding.
        self.place_at_grid(target_label, "B4", scale_factor=0.8)
        
        self.play(Create(scale_line))
        self.play(
            FadeIn(prediction_dot), Write(prediction_label),
            FadeIn(target_dot), Write(target_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Draw a double-headed arrow (#FFFFE0) between the two points and label it 'Distance'.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFFE0")
        
        distance_arrow = DoubleArrow(
            prediction_dot.get_center(), 
            target_dot.get_center(), 
            color="#FFFFE0", 
            buff=0.1,
            stroke_width=4
        )
        
        distance_label = Text("Distance", font_size=22, color="#FFFFE0")
        # Issue 27 fix: Adjusted scale_factor to 0.8 to prevent vertical crowding.
        self.place_at_grid(distance_label, "C4", scale_factor=0.8)
        
        self.play(GrowFromCenter(distance_arrow), Write(distance_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show the formula 'Error = Prediction - Target' in #FFFFFF at the top of the screen.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(WHITE)
        
        formula = MathTex(r"\text{Error} = \text{Prediction} - \text{Target}", color=WHITE)
        # Issue 26 fix: Adjusted scale_factor to 0.8 for better margins.
        self.place_in_area(formula, "A2", "A5", scale_factor=0.8)
        
        # We use updaters to ensure elements follow the prediction dot during the 'High loss' demonstration
        # Position labels 1 grid unit to the right of the vertical scale
        prediction_label.add_updater(lambda m: m.move_to(prediction_dot.get_center() + RIGHT * 1.0))
        
        distance_arrow.add_updater(lambda m: m.become(DoubleArrow(
            prediction_dot.get_center(), 
            target_dot.get_center(), 
            color="#FFFFE0", 
            buff=0.1,
            stroke_width=4
        )))
        distance_label.add_updater(lambda m: m.move_to((prediction_dot.get_center() + target_dot.get_center())/2 + RIGHT * 1.0))

        self.play(Write(formula))
        self.wait(1)
        
        # Increase distance to demonstrate "High loss"
        self.play(
            prediction_dot.animate.move_to(self.grid["F3"]),
            run_time=2
        )
        self.wait(2)
        
        # Cleanup
        prediction_label.clear_updaters()
        distance_arrow.clear_updaters()
        distance_label.clear_updaters()
