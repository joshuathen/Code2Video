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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Synthesis & Real-world Application", [
            "Synthesizing: Prior, Evidence, and Posterior probability.", 
            "Bayes' updates certainty in the face of uncertainty.", 
            "Testing example: Prior prevalence informs posterior diagnosis."
        ])
        
        # Load assets
        stethoscope = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/stethoscope.svg")
        chart = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/chart.svg")
        patient = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/patient.svg")
        
        # === Animation for Lecture Line 1 ===
        header = Text("Medical Diagnosis Example", font_size=32, color=WHITE)
        self.place_at_grid(header, 'A6', scale_factor=0.9)
        self.place_at_grid(stethoscope, 'B6', scale_factor=0.5)
        self.play(FadeIn(header), FadeIn(stethoscope))
        self.play(self.lecture[0].animate.set_color("#87CEEB"))

        # === Animation for Lecture Line 2 ===
        data = VGroup(
            MathTex(r"P(\text{Pos}|\text{Sick}) = 0.95", color="#FFD700"),
            MathTex(r"P(\text{Pos}|\text{Healthy}) = 0.05", color="#FF7F50")
        ).arrange(DOWN, aligned_edge=LEFT)
        self.place_in_area(data, 'B4', 'C6', scale_factor=0.7)
        self.place_at_grid(chart, 'D6', scale_factor=0.5)
        self.play(Write(data), FadeIn(chart))
        self.play(self.lecture[1].animate.set_color("#87CEEB"))

        # === Animation for Lecture Line 3 ===
        result = MathTex(r"P(\text{Sick}|\text{Pos}) \approx 0.16", color="#00FF00")
        self.place_at_grid(result, 'E5', scale_factor=1.0)
        self.place_at_grid(patient, 'F5', scale_factor=0.5)
        self.play(Create(result), FadeIn(patient))
        self.play(self.lecture[2].animate.set_color("#87CEEB"))
        self.wait(2)
