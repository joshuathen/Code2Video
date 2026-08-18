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
            "Transmission happens when infectious meet susceptible individuals.",
            "Recovery occurs as infectious individuals eventually recover.",
            "Mathematical rates define these transition rules.",
            "Blue dots turn red upon contact.",
            "Red dots slowly transition to green."
        ]
        self.setup_layout("The Engine: Mathematical Transition Rules", lecture_lines)
        
        # Equations
        eq1 = MathTex(r"dS/dt = -\beta SI/N").set_color("#FFFF00")
        eq2 = MathTex(r"dI/dt = \beta SI/N - \gamma I").set_color("#FF0000")
        eq3 = MathTex(r"dR/dt = \gamma I").set_color("#00FF00")
        
        # Assets
        human = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/human.svg")
        microbe = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microbe.svg")
        patient = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/patient.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.place_in_area(eq1, 'B2', 'B5', scale_factor=0.8)
        self.play(Write(eq1))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        self.place_in_area(eq2, 'C2', 'C5', scale_factor=0.8)
        self.play(Write(eq2))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.place_in_area(eq3, 'D2', 'D5', scale_factor=0.8)
        self.play(Write(eq3))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#ADD8E6"))
        self.place_at_grid(human, 'E2', scale_factor=0.5)
        self.place_at_grid(microbe, 'E4', scale_factor=0.5)
        self.play(FadeIn(human), FadeIn(microbe))
        self.play(human.animate.move_to(microbe.get_center()), run_time=0.5)
        self.play(Flash(microbe.get_center(), color=WHITE))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#90EE90"))
        self.place_at_grid(patient, 'F3', scale_factor=0.5)
        self.play(FadeIn(patient))
        self.play(patient.animate.set_color("#00FF00"), run_time=2)
        self.wait(2)
