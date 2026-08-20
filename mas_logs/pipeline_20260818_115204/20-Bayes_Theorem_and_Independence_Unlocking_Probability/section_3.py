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
        lecture_lines = [
            "Bayes' Theorem reverses conditional probability.",
            "Formula relates P(A|B) to P(B|A) and P(A).",
            "Use it to update beliefs with new evidence.",
            "Start from cause, move to effect.",
            "Essential for diagnostic reasoning."
        ]
        self.setup_layout("Introducing Bayes' Theorem", lecture_lines)
        
        # Colors
        color_1 = "#FF9999" # Light Red
        color_2 = "#99FF99" # Light Green
        color_3 = "#9999FF" # Light Blue
        color_4 = "#FFFF99" # Light Yellow
        
        # Define objects
        formula = MathTex(r"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}", font_size=48)
        self.place_in_area(formula, 'A2', 'C5', scale_factor=0.6)
        
        doctor_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/doctor.svg")
        patient_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/patient.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(formula))
        self.play(self.lecture[0].animate.set_color(color_1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(color_2))
        
        # Place icons near likelihood and prior components
        self.place_at_grid(doctor_icon, 'C3', scale_factor=0.3)
        self.place_at_grid(patient_icon, 'C5', scale_factor=0.3)
        
        self.play(FadeIn(doctor_icon), FadeIn(patient_icon))
        self.play(Indicate(formula[0][8:13])) # P(B|A)
        self.play(Indicate(formula[0][15:18])) # P(A)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(color_3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(color_4))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.play(formula.animate.scale(1.2))
        self.play(Indicate(formula))
        self.wait(2)
