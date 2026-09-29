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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Introduction: The 'Error' Concept", [
            "Neural networks learn by minimizing prediction errors.",
            "The Cost Function acts as a mathematical scorecard.",
            "It quantifies the gap between prediction and truth."
        ])
        
        # Elements
        error_label = Text("Error", font_size=36, color=WHITE)
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/scorecard.svg]
        scorecard_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scorecard.svg", color=WHITE)
        scorecard_label = VGroup(scorecard_icon, Text("Scorecard\n(Cost Function)", font_size=36, color=WHITE).arrange(DOWN))
        
        pred_circle = Circle(radius=0.3, color="#00FF00", fill_opacity=0.5)
        truth_circle = Circle(radius=0.3, color="#00FF00", fill_opacity=0.5)
        pred_text = Text("Prediction", font_size=20, color=WHITE).next_to(pred_circle, UP)
        truth_text = Text("Truth", font_size=20, color=WHITE).next_to(truth_circle, UP)
        
        prediction_group = VGroup(pred_circle, pred_text)
        truth_group = VGroup(truth_circle, truth_text)
        
        penalty_line = Line(start=pred_circle.get_center(), end=truth_circle.get_center(), color="#FF0000")
        penalty_label = Text("Penalty", font_size=24, color="#FF0000").next_to(penalty_line, RIGHT)
        penalty_element = VGroup(penalty_line, penalty_label)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        self.place_at_grid(error_label, 'C2', scale_factor=0.8)
        self.place_in_area(scorecard_label, 'C4', 'C6', scale_factor=0.7)
        self.play(Write(error_label), Write(scorecard_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.place_at_grid(prediction_group, 'D2', scale_factor=0.8)
        self.place_at_grid(truth_group, 'D5', scale_factor=0.8)
        self.play(Create(prediction_group), Create(truth_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.place_in_area(penalty_element, 'E2', 'E5', scale_factor=0.7)
        self.play(Create(penalty_line), Write(penalty_label))
        self.wait(2)
