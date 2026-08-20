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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Synthesis & Snell's Law", [
            "Substitute velocity with refractive indices.",
            "Arrive at Snell's Law equation.",
            "Nature optimizes for time efficiency."
        ])
        
        light_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/light.svg")

        # === Animation for Lecture Line 1 ===
        line1_eq = MathTex(r"v = \frac{c}{n}", color=WHITE)
        self.place_at_grid(line1_eq, "B2", scale_factor=1.5)
        light_1 = self.place_at_grid(light_icon.copy(), "A2", scale_factor=0.5)
        self.play(Write(line1_eq), FadeIn(light_1))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        sub_eq = MathTex(r"n_1 \sin(\theta_1) = n_2 \sin(\theta_2)", color="#00FF00")
        self.place_at_grid(sub_eq, "D3", scale_factor=1.2)
        light_2 = self.place_at_grid(light_icon.copy(), "C3", scale_factor=0.5)
        self.play(
            FadeOut(line1_eq), 
            FadeOut(light_1),
            Write(sub_eq),
            FadeIn(light_2)
        )
        self.lecture[1].set_color("#00FF00")
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        nature_text = Text("Nature minimizes Time", font_size=36, color=YELLOW)
        self.place_at_grid(nature_text, "E4", scale_factor=0.8)
        self.play(FadeIn(nature_text))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
