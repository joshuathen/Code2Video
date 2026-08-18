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
        self.setup_layout("Mathematical Mechanics: The Transition Equations", [
            "Transmission happens through interactions between populations.",
            "Infections occur as susceptible meet infected individuals.",
            "Recovery flows from infected to removed states.",
            "We model these using differential transition equations.",
            "Equations show how infection and recovery change."
        ])
        
        # Assets
        person = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/person.svg")
        microbe = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microbe.svg")
        hospital = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hospital.svg")
        syringe = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/syringe.svg")

        # Equations
        eq_ds = MathTex(r"dS/dt = -\beta SI/N", color=WHITE)
        eq_di = MathTex(r"dI/dt = \beta SI/N - \gamma I", color=WHITE)
        eq_dr = MathTex(r"dR/dt = \gamma I", color=WHITE)
        equations = VGroup(eq_ds, eq_di, eq_dr).arrange(DOWN, aligned_edge=LEFT)
        
        # Apply positioning constraints
        self.place_in_area(equations, 'A2', 'C4', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(eq_ds), FadeIn(self.place_at_grid(person, 'D2', 0.5)))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(eq_di), FadeIn(self.place_at_grid(microbe, 'D4', 0.5)))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(eq_dr), FadeIn(self.place_at_grid(hospital, 'F2', 0.5)))
        self.lecture[2].set_color(YELLOW)

        # === Animation for Lecture Line 4 ===
        # Highlight infection term
        inf_term = eq_di[0][0:7]
        self.play(Indicate(inf_term, color=RED), FadeIn(self.place_at_grid(microbe.copy(), 'E4', 0.3)))
        self.lecture[3].set_color(YELLOW)

        # === Animation for Lecture Line 5 ===
        # Highlight recovery term
        rec_term = eq_di[0][8:12]
        self.play(Indicate(rec_term, color=GREEN), FadeIn(self.place_at_grid(syringe, 'E6', 0.5)))
        self.lecture[4].set_color(YELLOW)
        
        self.wait(2)
