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
        lecture_lines = [
            "Diseases spread through population interactions.", 
            "Individuals move between three distinct states.", 
            "Susceptible individuals can become infected.", 
            "Infected individuals eventually recover or gain immunity.", 
            "[Asset: SIR_Flow_Diagram] depicts these dynamic transitions."
        ]
        self.setup_layout("The Hook: Why do diseases spread?", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Population Cloud: Using asset icon/human.svg
        dots = VGroup(*[SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/human.svg", color=WHITE) for _ in range(12)])
        self.place_in_area(dots, 'A1', 'C6', scale_factor=0.3)
        self.play(FadeIn(dots))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Patient Zero: Using asset icon/virus.svg
        virus = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/virus.svg", color="#FF0000")
        virus.move_to(dots[6].get_center())
        self.play(FadeIn(virus), dots[6].animate.set_color("#FF0000"), run_time=0.5)
        self.lecture[2].set_color("#FF5555")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Transmission simulation
        animations = []
        for i in range(len(dots)):
            if i != 6:
                animations.append(dots[i].animate.set_color("#FF5555"))
        self.play(*animations, run_time=1.5)
        self.lecture[3].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Show SIR Flow Diagram
        sir_diagram = VGroup(
            Rectangle(width=3, height=1.5, color=BLUE),
            Text("S -> I -> R", font_size=24, color=WHITE)
        ).arrange(DOWN)
        self.place_in_area(sir_diagram, 'D1', 'F6', scale_factor=0.6)
        self.play(FadeIn(sir_diagram))
        self.lecture[4].set_color(PURPLE)
        self.wait(2)
