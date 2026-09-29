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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Dynamics of Transmission", [
            "Transmission rate beta drives new infections.",
            "Infection happens through contact between Susceptible and Infectious.",
            "Higher beta means faster disease spread."
        ])
        
        # Assets
        person_s = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/person.svg", color=BLUE)
        person_i = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/person.svg", color=RED)
        virus = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/virus.svg", color=WHITE)
        
        s_label = Text("S", font_size=24, color=BLUE)
        i_label = Text("I", font_size=24, color=RED)
        
        self.place_at_grid(person_s, "C2", scale_factor=0.3)
        self.place_at_grid(s_label, "C1", scale_factor=0.8)
        self.place_at_grid(person_i, "C5", scale_factor=0.3)
        self.place_at_grid(i_label, "C6", scale_factor=0.8)
        
        beta_text = MathTex(r"\beta", color=YELLOW)
        self.place_at_grid(beta_text, "E3", scale_factor=1.5)
        
        self.place_at_grid(virus, "B3", scale_factor=0.3)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(beta_text))
        # Virus pulses
        self.play(Indicate(virus, color=WHITE))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        # Particles moving between S and I
        path = CurvedArrow(self.grid["C2"], self.grid["C5"], angle=-PI/4, color=WHITE)
        self.play(Create(path))
        self.play(person_s.animate.move_to(self.grid["C4"]), run_time=1)
        self.play(person_s.animate.move_to(self.grid["C2"]), run_time=1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(beta_text.animate.scale(2.0))
        self.play(FadeOut(path))
