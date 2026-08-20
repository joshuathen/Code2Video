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
        lecture_lines = ["Turbulence is a multi-scale energy cascade.", "Reynolds Number defines the transition to chaos.", "Predictable laminar flow breaks down."]
        self.setup_layout("The Hook: The Chaos of the Coffee Cup", lecture_lines)
        
        # Animations
        # === Animation for Lecture Line 1 ===
        # Show a swirling [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cup.svg] model, label it 'Turbulent Flow' in #FF00FF
        cup = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cup.svg", color=BLUE)
        self.place_at_grid(cup, 'D4', scale_factor=0.6)
        label1 = Text("Turbulent Flow", font_size=24, color="#FF00FF")
        self.place_at_grid(label1, 'B4', scale_factor=0.7)
        self.play(FadeIn(cup), Write(label1))
        self.play(self.lecture[0].animate.set_color("#FF00FF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Fade out the cup, show a grid of velocity vectors in #00FFFF
        self.play(FadeOut(cup), FadeOut(label1))
        grid = VGroup(*[Arrow(start=ORIGIN, end=RIGHT*0.3, color=WHITE).shift(i*RIGHT*0.5 + j*UP*0.5) for i in range(-2, 3) for j in range(-2, 3)])
        self.place_in_area(grid, 'D2', 'F5', scale_factor=0.6)
        self.play(Create(grid))
        self.play(grid.animate.set_color("#00FFFF"))
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show small [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/coffee.svg] eddies emerging, highlight them in #FFFF00
        eddies = VGroup(*[SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coffee.svg", color="#FFFF00").scale(0.3).move_to(self.grid['C3'] + np.array([0.2*i, 0.2*j, 0])) for i in range(-1, 2) for j in range(-1, 2)])
        self.play(FadeIn(eddies))
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.wait(2)
