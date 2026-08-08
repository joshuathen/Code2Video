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
            "Increasing mass ratios change the collision count.",
            "Mass ratio 100 yields 31 collisions.",
            "Mass ratio 10,000 yields 314 collisions.",
            "The count converges to digits of Pi.",
            "This links simple physics to transcendental numbers."
        ]
        self.setup_layout("The Mathematical Bridge", lecture_lines)
        
        # Assets
        billiard_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/billiard.svg")
        blocks_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF6666"))
        self.place_at_grid(billiard_icon, "A3", scale_factor=0.5)
        label_r = Text("R", color="#00FFFF").next_to(billiard_icon, RIGHT)
        formula = MathTex(r"N \approx \pi \sqrt{\frac{M_1}{M_2}}").scale(0.8)
        self.place_in_area(formula, "A2", "B5", scale_factor=0.8)
        self.play(FadeIn(billiard_icon), Write(label_r), Write(formula))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#66FF66"))
        self.place_at_grid(blocks_icon, "C3", scale_factor=0.5)
        count_31 = Text("31", color=WHITE).next_to(blocks_icon, RIGHT)
        self.play(FadeIn(blocks_icon), Write(count_31))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#6666FF"))
        count_314 = Text("314", color=WHITE).next_to(blocks_icon, RIGHT)
        self.play(Transform(count_31, count_314))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFF66"))
        pi_text = Text(r"$\approx 3.14$", color="#FF00FF").scale(1.2)
        self.place_in_area(pi_text, "C2", "D5", scale_factor=0.8)
        self.play(ReplacementTransform(formula, pi_text))
        self.play(pi_text.animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF66FF"))
        self.wait(2)
