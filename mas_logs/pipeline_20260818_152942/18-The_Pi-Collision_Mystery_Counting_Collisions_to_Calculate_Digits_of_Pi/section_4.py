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
        lecture_lines = [
            "Bounces follow the arc of a circle.",
            "Reflections equate to Pi's geometric ratio.",
            "Mass ratios determine the number of digits.",
            "Pi emerges from these simple physical dynamics.",
            "Higher mass ratios reveal more Pi digits."
        ]
        self.setup_layout("The Emergence of Pi", lecture_lines)
        
        # Objects (Assets)
        # Using SVGMobject for SVG files
        circle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color=YELLOW)
        block_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=BLUE)
        
        diameter = Line(start=LEFT, end=RIGHT, color=BLUE)
        pi_text = MathTex(r"\pi \approx 3.14159...", color=WHITE)
        label_ratio = Text("Mass Ratio: 100", font_size=20, color=GREEN)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_in_area(circle_icon, "A4", "C6", scale_factor=0.6)
        self.play(FadeIn(circle_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        self.place_at_grid(diameter, "B5", scale_factor=0.8)
        self.play(Create(diameter))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.place_at_grid(label_ratio, "E2", scale_factor=0.8)
        self.play(Write(label_ratio))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(RED))
        self.place_at_grid(pi_text, "E4", scale_factor=0.7)
        self.play(Write(pi_text))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(PURPLE))
        self.play(circle_icon.animate.scale(1.2), pi_text.animate.set_color(PURPLE))
        self.wait(1)
