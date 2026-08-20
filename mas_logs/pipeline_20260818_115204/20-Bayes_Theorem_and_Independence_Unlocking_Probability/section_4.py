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
        self.setup_layout("Synthesis: Applying Bayes in Real Scenarios", [
            "Bayes applies differently based on event independence.",
            "Independent events simplify the calculation significantly.",
            "Naive Bayes leverages this independence assumption.",
            "Dependent events require the full formula.",
            "Real world scenarios often mix these structures."
        ])
        
        # Assets
        patient_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/patient.svg")
        chart_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/chart.svg")
        
        # Elements
        box_indep = Rectangle(color=BLUE, height=1.5, width=2.5)
        text_indep = Text("P(A|B) = P(A)", font_size=20, color=BLUE)
        group_indep = VGroup(box_indep, text_indep)
        
        box_dep = Rectangle(color=RED, height=1.5, width=2.5)
        text_dep = Text("P(A|B) = P(B|A)P(A)/P(B)", font_size=16, color=RED)
        group_dep = VGroup(box_dep, text_dep)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(patient_icon, "B2", 0.5)
        self.play(FadeIn(patient_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        # Fix: Move to A4-B6 area as per critic
        self.place_in_area(group_indep, "A4", "B6", scale_factor=0.6)
        self.play(FadeIn(group_indep))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(Indicate(group_indep))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        # Fix: Move to A4-B6 area as per critic
        self.place_in_area(group_dep, "A4", "B6", scale_factor=0.6)
        self.play(FadeIn(group_dep))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        self.play(FadeOut(group_indep), FadeOut(group_dep), FadeOut(patient_icon))
        
        summary = Text("Bayes: The Engine of Inference", font_size=24, color=GREEN)
        # Fix: Move as per critic
        self.place_in_area(summary, "C4", "D6", scale_factor=0.8)
        
        self.place_at_grid(chart_icon, "E3", 0.8)
        self.play(FadeIn(summary), FadeIn(chart_icon))
        self.wait(2)
