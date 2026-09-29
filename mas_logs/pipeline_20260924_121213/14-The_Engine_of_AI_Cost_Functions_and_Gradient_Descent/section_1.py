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
        self.setup_layout("Prerequisite: The Prediction Gap", [
            "Neural networks are like students guessing answers.",
            "We compare the prediction to ground truth.",
            "The loss is the distance between them."
        ])
        
        # Load assets
        student_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/student.svg")
        pred_label = Text("Prediction", color=BLUE, font_size=24)
        actual_label = Text("Actual", color=GREEN, font_size=24)
        
        pred_dot = Dot(color=BLUE)
        actual_dot = Dot(color=GREEN)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(student_icon, 'B3', scale_factor=0.5)
        self.place_at_grid(pred_label, 'C2', scale_factor=0.8)
        self.place_at_grid(actual_label, 'C5', scale_factor=0.8)
        self.play(FadeIn(student_icon), FadeIn(pred_label), FadeIn(actual_label))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.place_at_grid(pred_dot, 'D2', scale_factor=1.0)
        self.place_at_grid(actual_dot, 'D5', scale_factor=1.0)
        self.play(FadeIn(pred_dot), FadeIn(actual_dot))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        error_line = Line(pred_dot.get_center(), actual_dot.get_center(), color="#FF00FF")
        error_label = MathTex(r"|P - A|", color="#FF00FF", font_size=24)
        self.place_at_grid(error_label, 'D4', scale_factor=0.9)
        
        self.play(Create(error_line), Write(error_label))
        self.wait(2)
