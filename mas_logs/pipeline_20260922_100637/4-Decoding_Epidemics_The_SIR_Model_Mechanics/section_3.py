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
        self.setup_layout("The Recovery Mechanism (The Gamma Factor)", [
            "Gamma represents the rate of recovery.",
            "Infected individuals transition to the Recovered state.",
            "The infectious period ends as Gamma takes effect."
        ])
        
        # Elements
        gamma_text = Text("Gamma (γ)", color="#00CED1", font_size=32)
        self.place_at_grid(gamma_text, "C3", scale_factor=0.8)
        
        # Assets
        infected_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/person.svg").set_height(0.6)
        recovered_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/patient.svg").set_height(0.6)
        arrow = Arrow(start=LEFT, end=RIGHT, color=WHITE).set_length(1.0)
        
        transition_group = VGroup(infected_icon, arrow, recovered_icon).arrange(RIGHT)
        self.place_at_grid(transition_group, "C4", scale_factor=0.9)
        
        recovery_label = Text("Recovery", color=GREEN, font_size=24)
        self.place_at_grid(recovery_label, "C5", scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00CED1"), Write(gamma_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF4500"), FadeIn(transition_group), Write(recovery_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#32CD32"))
        
        # Animation of transition
        self.play(
            infected_icon.animate.set_opacity(0.3),
            FadeIn(recovered_icon),
            run_time=2
        )
        self.wait(2)
