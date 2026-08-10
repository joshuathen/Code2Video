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
        self.setup_layout("Introduction to Bayes' Theorem", [
            "Bayes' Theorem reverses conditional probability to update knowledge.",
            "Flow chart maps Prior probabilities to Posterior beliefs.",
            "The formula: P(A|B) equals P(B|A) times P(A) over P(B).",
            "We use this to infer causes from observed effects.",
            "Testing for rare diseases makes this vital."
        ])
        
        # Load assets
        icon_patient = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/patient.svg")
        icon_syringe = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/syringe.svg")
        icon_chart = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/chart.svg")
        
        # === Animation for Lecture Line 1 ===
        bayes_formula = MathTex(
            r"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}",
            font_size=42,
            tex_template=TexTemplate()
        )
        self.place_in_area(bayes_formula, "B1", "C4", scale_factor=0.65)
        self.play(Write(bayes_formula))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        flow_chart = VGroup(
            Text("Prior", font_size=24),
            Arrow(LEFT, RIGHT),
            Text("Evidence", font_size=24),
            Arrow(LEFT, RIGHT),
            Text("Posterior", font_size=24)
        ).arrange(RIGHT, buff=0.2)
        self.place_at_grid(flow_chart, "D3", scale_factor=0.7)
        self.play(FadeIn(flow_chart))
        flow_chart.set_color("#00FFFF")
        self.lecture[1].set_color(BLUE)

        # === Animation for Lecture Line 3 ===
        self.play(
            bayes_formula[0][4:10].animate.set_color("#FF5733"), 
            bayes_formula[0][11:13].animate.set_color("#33FF57")
        )
        self.lecture[2].set_color(BLUE)

        # === Animation for Lecture Line 4 ===
        icon_patient.set_color("#FF0000")
        icon_syringe.set_color("#FF0000")
        self.place_at_grid(icon_patient, "E2", scale_factor=0.5)
        self.place_at_grid(icon_syringe, "E4", scale_factor=0.5)
        arrow = Arrow(icon_patient.get_right(), icon_syringe.get_left(), color="#FF0000")
        self.play(FadeIn(icon_patient), FadeIn(icon_syringe), Create(arrow))
        self.lecture[3].set_color(BLUE)

        # === Animation for Lecture Line 5 ===
        icon_chart.set_color("#00FF00")
        self.place_at_grid(icon_chart, "B5", scale_factor=0.6)
        self.play(FadeIn(icon_chart))
        self.lecture[4].set_color(BLUE)
        self.wait(2)
