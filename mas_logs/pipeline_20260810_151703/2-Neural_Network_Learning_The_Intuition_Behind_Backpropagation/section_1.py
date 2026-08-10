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
        lines = [
            "Neural networks are like tuning knobs on a machine.",
            "Goal: Minimize the difference between prediction and truth.",
            "Analogy: A student learning to throw paper airplanes.",
            "Missed target indicates error, requiring adjustments.",
            "Adjust arm angle to hit the target eventually."
        ]
        self.setup_layout("The Analogy: Learning by Correction", lines)
        
        # Assets
        student = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/student.svg")
        airplane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/airplane.svg")
        knob = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/knob.svg")
        target = Square(side_length=0.4, color=GREEN).rotate(PI/4)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        knob.set_color("#44BBFF")
        self.place_at_grid(knob, 'B2', 0.8)
        self.play(FadeIn(knob))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(RED)
        pred_label = Text("Prediction", font_size=20, color=BLUE)
        truth_label = Text("Truth", font_size=20, color=RED)
        self.place_at_grid(pred_label, 'A5', 0.8)
        self.place_at_grid(truth_label, 'C5', 0.8)
        self.play(FadeIn(pred_label), FadeIn(truth_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        self.place_at_grid(student, 'E2', 0.8)
        self.place_at_grid(airplane, 'E3', 0.5)
        self.place_at_grid(target, 'E5', 0.8)
        self.play(FadeIn(student), FadeIn(airplane), FadeIn(target))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(ORANGE)
        error_line = Line(airplane.get_center(), target.get_center(), color="#FF4444", stroke_width=2)
        error_text = Text("Error!", font_size=20, color=WHITE)
        self.place_at_grid(error_text, 'D2', 0.9)
        self.play(Create(error_line), Write(error_text))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(GREEN)
        self.play(Rotate(knob, angle=PI/4), student.animate.set_color(WHITE))
        self.play(FadeOut(error_line), FadeOut(error_text))
        self.play(airplane.animate.move_to(target.get_center()))
        flash = Text("Correction!", font_size=24, color=WHITE)
        self.place_at_grid(flash, 'D5', 1.0)
        self.play(FadeIn(flash))
