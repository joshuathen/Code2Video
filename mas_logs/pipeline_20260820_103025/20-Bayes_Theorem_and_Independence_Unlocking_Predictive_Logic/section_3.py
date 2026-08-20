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
            "Bayes' theorem updates beliefs with new evidence.",
            "Formula: Posterior equals Likelihood times Prior over Evidence.",
            "It refines probability based on observed data.",
            "Example: Sensor detections adjust disease likelihood.",
            "It is the core of predictive logic."
        ]
        self.setup_layout("Introduction to Bayes' Theorem", lecture_lines)
        
        # Set all lecture lines to visible
        self.lecture.set_opacity(1)

        # Animation setup using assets
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg
        sensor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg", color=BLUE)
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/disease.svg
        disease = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disease.svg", color=RED)
        
        # Branching elements
        root = Dot(color=BLUE)
        event_a = Dot(color=GREEN)
        event_not_a = Dot(color=RED)

        # Applying grid positioning requirements (Issue 27, 38)
        self.place_at_grid(root, 'B3', scale_factor=0.6)
        self.place_at_grid(event_a, 'C2', scale_factor=0.6)
        self.place_at_grid(event_not_a, 'C4', scale_factor=0.6)
        
        line1 = Line(root.get_center(), event_a.get_center(), color=WHITE)
        line2 = Line(root.get_center(), event_not_a.get_center(), color=WHITE)
        
        # Labels
        label_a = MathTex("A", color=GREEN).scale(0.7).next_to(event_a, UP)
        label_not_a = MathTex("\\neg A", color=RED).scale(0.7).next_to(event_not_a, UP)
        formula = MathTex("P(A|B)", color=YELLOW)
        group_label = Text("Evidence", color=WHITE)
        
        # Applying area positioning requirements (Issue 28, 29, 38)
        self.place_in_area(formula, 'B5', 'C6', scale_factor=0.8)
        self.place_in_area(group_label, 'D3', 'D4', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(FadeIn(root), FadeIn(sensor.next_to(root, UP, buff=0.1)))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.play(Create(line1), Create(line2), Write(label_a), Write(label_not_a), FadeIn(formula))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        self.play(FadeIn(disease.next_to(event_a, DOWN, buff=0.1)))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(RED)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(PURPLE)
        self.play(Write(group_label))
